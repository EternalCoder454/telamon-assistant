//! The model: llama-server's OpenAI-compatible chat API (telamon-llama, run
//! with `--jinja` so tool calls work), with the read-only tools in a loop.

use crate::tools;
use anyhow::{Context as _, Result, anyhow};
use serde_json::{Value, json};
use std::time::Duration;

/// Tool rounds before the model must answer.
const MAX_ROUNDS: usize = 4;

const SYSTEM: &str = "You are Telamon, the voice assistant of Telamon OS, \
speaking with the computer's owner. Your answers are read aloud, so: answer in \
one to three short sentences, plain spoken English, no lists, no markdown, no \
emoji, no URLs. Round numbers and say units in words (percent, degrees, \
gigabytes). Use the tools for anything about this computer, the time, the \
date, files, the network, the location or the weather; never guess those. \
The tools can only look, never change anything: if asked to do something, say \
you can't do that yet. If the question is unclear, ask one short question.";

pub struct Llm {
    url: String,
    agent: ureq::Agent,
    pub tools: tools::Context,
}

/// One answer, and the tools it used.
pub struct Answer {
    pub text: String,
    pub tools: Vec<String>,
}

impl Llm {
    pub fn new(url: &str, tools: tools::Context) -> Self {
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(60)))
            .build()
            .into();
        Self { url: url.trim_end_matches('/').to_string(), agent, tools }
    }

    /// Whether llama-server answers its health check.
    pub fn ready(&self) -> bool {
        self.agent
            .get(format!("{}/health", self.url))
            .call()
            .is_ok()
    }

    /// The answer to `question`; `on_tool` hears each tool as it runs.
    pub fn ask(&self, question: &str, on_tool: &mut dyn FnMut(&str)) -> Result<Answer> {
        let mut messages = vec![
            json!({"role": "system", "content": SYSTEM}),
            json!({"role": "user", "content": question}),
        ];
        let mut used = Vec::new();
        for round in 0..=MAX_ROUNDS {
            let mut body = json!({
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 220,
                // Thinking costs 2-4x the latency (measured in Telamon Gates).
                "reasoning_effort": "low",
                "chat_template_kwargs": {"enable_thinking": false},
            });
            if round < MAX_ROUNDS {
                body["tools"] = tools::definitions();
            }
            let reply: Value = self
                .agent
                .post(format!("{}/v1/chat/completions", self.url))
                .send_json(&body)
                .context("llama-server")?
                .body_mut()
                .read_json()
                .context("llama-server's reply")?;
            let message = reply["choices"][0]["message"].clone();
            let calls = message["tool_calls"].as_array().cloned().unwrap_or_default();
            if calls.is_empty() {
                let text = message["content"].as_str().unwrap_or("").to_string();
                return Ok(Answer { text: spoken(&text), tools: used });
            }
            messages.push(json!({
                "role": "assistant",
                "content": message["content"].as_str().unwrap_or(""),
                "tool_calls": calls,
            }));
            for call in &calls {
                let name = call["function"]["name"].as_str().unwrap_or("");
                let args = call["function"]["arguments"].as_str().unwrap_or("{}");
                on_tool(name);
                log::info!("tool {name} {args}");
                let result = tools::call(&self.tools, name, args);
                used.push(name.to_string());
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": call["id"].as_str().unwrap_or(""),
                    "content": result,
                }));
            }
        }
        Err(anyhow!("the model kept calling tools"))
    }
}

/// The reply as something to say: no markdown, no thinking.
pub fn spoken(text: &str) -> String {
    let text = match text.rfind("</think>") {
        Some(i) => &text[i + "</think>".len()..],
        None => text,
    };
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        let line = line.trim().trim_start_matches(['#', '>', '-', '*']).trim();
        if line.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(line);
    }
    out.replace(['*', '`', '_'], "")
}

/// `text` cut into sentences, so speech can start after the first.
pub fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = text.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        cur.push(c);
        let end = matches!(c, '.' | '!' | '?' | ';')
            && chars.get(i + 1).is_none_or(|n| n.is_whitespace())
            // "3.5" or "e.g." mid-sentence: only end before a capital or the end.
            && chars
                .iter()
                .skip(i + 1)
                .find(|n| !n.is_whitespace())
                .is_none_or(|n| n.is_uppercase() || n.is_ascii_digit());
        if end && cur.trim().len() > 1 {
            out.push(cur.trim().to_string());
            cur.clear();
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speaks_plain_text() {
        assert_eq!(spoken("<think>x</think>\n**Hi** there"), "Hi there");
        assert_eq!(spoken("- one\n- two"), "one two");
    }

    #[test]
    fn splits_sentences() {
        assert_eq!(
            sentences("It's 3.5 degrees. Cold! Wear a coat"),
            vec!["It's 3.5 degrees.", "Cold!", "Wear a coat"]
        );
        assert_eq!(sentences("Hello."), vec!["Hello."]);
    }
}
