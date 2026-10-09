"""Text inventory for synthetic wake-word data (all deterministic).

Wake phrases are given as (text, phonemes_or_None). Kokoro (via espeak-ng)
says "Telamon" as TEL-uh-mon; extra phoneme-level variants widen coverage.
"""

from __future__ import annotations

import random

# ---------------------------------------------------------------------------
# Positives
# ---------------------------------------------------------------------------
PREFIXES = ["", "Hey ", "Hey, ", "Okay ", "OK ", "Hey there, ", "Hi "]
# Spelling variants that change the pronunciation (all intended to be heard as Telamon).
NAME_SPELLINGS = ["Telamon", "Tellamon", "Telamun", "Tel-a-mon", "Telemon", "Tellamun"]
SUFFIXES = ["", ",", "."]

# Hand-written IPA for the intended TEL-uh-mon, in the Kokoro vocabulary.
NAME_IPA = ["tˈɛləmən", "tˈɛləmˌɑn", "tˈɛləmˌɔn", "tˈɛlɐmən"]
PREFIX_IPA = {
    "": "",
    "Hey ": "hˈA ",
    "Hey, ": "hˈA, ",
    "Okay ": "ˌOkˈA ",
    "OK ": "ˌOkˈA ",
    "Hey there, ": "hˈA ðˈɛɹ, ",
    "Hi ": "hˈI ",
}


def wake_phrases() -> list[tuple[str, str | None]]:
    """(text, phonemes). phonemes is None -> let espeak phonemize `text`."""
    out: list[tuple[str, str | None]] = []
    # Core phrases requested by the brief, spelled normally.
    for t in ["Telamon", "Hey Telamon", "Hey, Telamon", "Okay Telamon", "Telamon,",
              "OK Telamon", "Hey Telamon.", "Hi Telamon"]:
        out.append((t, None))
    # Spelling variants that shift the pronunciation.
    for name in NAME_SPELLINGS[1:]:
        for pre in ["", "Hey ", "Hey, ", "Okay "]:
            out.append((f"{pre}{name}", None))
    # Explicit IPA for TEL-uh-mon.
    for pre, ipre in PREFIX_IPA.items():
        for ipa in NAME_IPA:
            out.append((f"{pre}Telamon [ipa]", f"{ipre}{ipa}"))
            out.append((f"{pre}Telamon, [ipa]", f"{ipre}{ipa},"))
    seen, uniq = set(), []
    for t, p in out:
        if (t, p) not in seen:
            seen.add((t, p))
            uniq.append((t, p))
    return uniq


# ---------------------------------------------------------------------------
# Confusable negatives
# ---------------------------------------------------------------------------
CONFUSABLES = [
    "telephone", "tell a man", "tell them on", "salmon", "Solomon", "lemon", "Tel Aviv",
    "telescope", "talisman", "Pokémon", "hey Thomas", "hey Siri", "tell him on Monday",
    "hey Tim", "hey Salomon", "hey Jarvis", "hey Alexa", "okay Google", "tell a mum",
    "tell her mom", "tele monitor", "tell me more", "Tell Emmen", "tell a mountain",
    "Tallahassee", "telegram", "telemetry", "telepathy", "television", "tell them all",
    "hey Dalton", "hey Damon", "hey Ramon", "hey Simon", "hey Lemmon", "hey Salmon",
    "hey Tilly", "hey Tammy", "hey tell me", "hey, tell a man",
    "Thelma", "Telus", "tell a mom", "the lemon",
    "Thomas", "Colossimo", "Telemachus",
    "Hey Mon", "Hey Telly", "Hey Tela", "Hey Mona", "Hey Lamar", "Okay Thomas",
    "Okay Google", "Ok Siri", "Hey computer", "Hey Cortana", "Hey Mycroft", "Hey Rhasspy",
    "tell them on the phone", "tell a man to call", "I will tell him on Monday",
]

# Near-homophones of the wake word. A detector cannot and should not separate
# these from "Telamon" (they sound the same to the embedding), so they are in
# neither the training negatives nor the evaluation: kept here for the record.
NEAR_HOMOPHONES = ["Telemann", "Teleman", "hey Teleman", "Tolomon", "Tel-mon", "Talmon", "Dalamon", "Delamon", "Tolman", "tell Amon"]

# ---------------------------------------------------------------------------
# Ordinary sentences (templated + hand written), deterministic, deduplicated
# ---------------------------------------------------------------------------
HAND = [
    "What time is it right now?", "How busy is my CPU right now?", "What's the weather like today?",
    "Set a timer for ten minutes.", "Play some relaxing music.", "How much memory is the browser using?",
    "Remind me to call my mother tomorrow.", "Turn the volume down a little.", "What is the capital of France?",
    "Can you open the downloads folder?", "Is the network connection working?",
    "The quick brown fox jumps over the lazy dog.", "She sells seashells by the seashore.",
    "I think we should leave before it starts raining.", "He walked slowly down the long, empty corridor.",
    "The meeting has been moved to three o'clock on Thursday.", "Please send me the report by the end of the day.",
    "My brother lives in a small town near the coast.", "There is a lot of traffic on the bridge this morning.",
    "We could order pizza tonight if nobody feels like cooking.", "The library closes at eight on weekdays.",
    "It was the best concert I have ever been to.", "Did you remember to lock the front door?",
    "Nobody knew where the keys had gone.", "The train to the airport leaves every twenty minutes.",
    "I would like a cup of coffee and a slice of toast.", "The old house at the end of the street is finally being repaired.",
    "Could you repeat that a little more slowly?", "What did the doctor say about your knee?",
    "In the autumn the leaves turn red and gold.", "They argued about the budget for most of the afternoon.",
    "The kids are playing in the garden until dinner.", "Her favorite subject at school was always history.",
    "Add milk, eggs and butter to the shopping list.", "How long does it take to boil an egg?",
    "The new update fixes several bugs and improves battery life.", "I forgot my umbrella on the bus again.",
    "Turn left at the next light and then go straight for two miles.", "He has been learning to play the piano for three years.",
    "The museum is open every day except Monday.", "She spoke so quietly that nobody heard her.",
    "Let's have lunch at noon and talk about the schedule.", "The movie was longer than I expected.",
    "A gentle breeze moved across the open field.", "Please remember to turn off the lights when you leave.",
    "Is there anything interesting on the news tonight?", "The soup needs a little more salt and some pepper.",
    "We watched the sun set over the mountains.", "My phone battery is almost empty.",
    "Where did you put the red notebook I gave you?", "It might snow later this week, according to the forecast.",
    "The cat jumped onto the table and knocked over a glass.", "He said he would be home by ten, but it is already midnight.",
    "Many people prefer tea in the morning.", "The bakery on the corner sells fresh bread every day.",
    "Two plus two equals four, and four plus four equals eight.", "Can you show me how much disk space is left?",
    "What are the top processes using memory?", "Read me the first line of that document.",
    "She called her friend to talk about the weekend plans.", "They painted the whole fence white last summer.",
    "The professor explained the theory in simple terms.", "Don't forget to water the plants before you go.",
    "Salmon is my favorite fish, especially grilled with lemon.", "Solomon was known for his wisdom.",
    "We visited Tel Aviv and Jerusalem on the same trip.", "He looked at the stars through his telescope.",
    "The telephone rang twice and then stopped.", "She wore a talisman around her neck.",
    "The kids collect Pokémon cards and trade them at school.", "Hey Thomas, are you coming to the game tonight?",
    "Hey Siri, what's the weather like in London?", "Tell him on Monday that the package arrived.",
    "I need to tell them on Friday about the changes.", "A lemon tree grew behind the old barn.",
    "Tell a man to bring more chairs for the guests.", "The television in the living room is broken.",
]

_SUBJ = ["the teacher", "my neighbor", "the engineer", "our guests", "the old man", "a young girl", "the committee",
         "my cousin", "the driver", "the whole team", "your sister", "the manager", "a stranger", "the baker",
         "the farmers", "my grandmother", "the doctor", "the students", "an artist", "the mayor"]
_VERB = ["carried", "forgot", "painted", "repaired", "borrowed", "discussed", "cleaned", "noticed", "finished",
         "ordered", "photographed", "packed", "described", "watched", "counted", "delivered", "tested", "sold"]
_OBJ = ["the heavy boxes", "a broken chair", "the garden gate", "an old map", "the morning paper", "three red apples",
        "the front window", "a small wooden boat", "the kitchen floor", "several large envelopes", "a handwritten letter",
        "the new schedule", "some fresh flowers", "the second bedroom", "a long speech"]
_WHEN = ["on Tuesday morning", "after lunch", "before the storm arrived", "late last night", "every single week",
         "just before sunrise", "during the holidays", "when nobody was looking", "as soon as possible", "in the spring"]
_PLACE = ["at the station", "in the village", "near the river", "behind the school", "across the street", "inside the warehouse",
          "by the lake", "on the third floor", "outside the cafe", "at the end of the road"]
_Q = ["What time does the {n} open?", "Where can I find a good {n}?", "How far is the nearest {n}?",
      "Is the {n} still available?", "Who owns that {n}?", "Why is the {n} so expensive?", "When will the {n} be ready?"]
_N = ["bakery", "pharmacy", "garage", "library", "restaurant", "market", "hotel", "gym", "theater", "bank", "museum", "laundry"]
_CMD = ["Open the {a} and show me the {b}.", "Search for {a} near {c}.", "Write a note about the {a}.",
        "Find the file named {a} in my {b}.", "How many {a} are in the {b}?", "Remind me about the {a} at {t}.",
        "Call {c} and ask about the {a}.", "Show me the latest {a} from {c}."]
_A = ["invoice", "photos", "recipe", "calendar", "report", "budget", "playlist", "contract", "schedule", "email", "tickets"]
_B = ["documents folder", "downloads", "desktop", "music library", "projects directory", "pictures folder"]
_C = ["Paris", "Denver", "my sister", "the office", "Toronto", "the dentist", "Berlin", "Sam", "Nairobi", "the bank"]
_T = ["nine thirty", "a quarter past two", "noon", "seven forty five", "half past four", "six o'clock"]


def sentence_pool(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    out: set[str] = set()
    guard = 0
    while len(out) < n and guard < n * 50:
        guard += 1
        k = rng.random()
        if k < 0.45:
            s = f"{rng.choice(_SUBJ)} {rng.choice(_VERB)} {rng.choice(_OBJ)} {rng.choice(_PLACE)} {rng.choice(_WHEN)}."
            s = s[0].upper() + s[1:]
        elif k < 0.65:
            s = rng.choice(_Q).format(n=rng.choice(_N))
        elif k < 0.9:
            s = rng.choice(_CMD).format(a=rng.choice(_A), b=rng.choice(_B), c=rng.choice(_C), t=rng.choice(_T))
        else:
            s = f"{rng.choice(_SUBJ)} {rng.choice(_VERB)} {rng.choice(_OBJ)}, and then {rng.choice(_SUBJ)} {rng.choice(_VERB)} {rng.choice(_OBJ)}."
            s = s[0].upper() + s[1:]
        out.add(s)
    return sorted(out)


def split_sentences(seed: int = 1234) -> dict[str, list[str]]:
    """Disjoint sentence pools: train / val / test."""
    pool = sorted(set(HAND) | set(sentence_pool(900, seed)))
    rng = random.Random(seed)
    rng.shuffle(pool)
    n = len(pool)
    test = pool[: n // 4]
    val = pool[n // 4 : n // 4 + n // 8]
    train = pool[n // 4 + n // 8 :]
    return {"train": train, "val": val, "test": test}


# Word-salad negatives: random dictionary words with function words mixed in.
# They are not meaningful sentences; they exist for broad phonetic coverage.
_FUNC = ("the a of and to in that it with for on as at by from or this but not are was be have you we they he she I can will would "
         "should about there their what which when who how all some one two three more no yes if so then than into out up down over "
         "after before again here very just also only your my our his her its").split()


def salad_pool(n: int, seed: int, dict_path: str = "/usr/share/dict/words") -> list[str]:
    try:
        raw = open(dict_path, encoding="utf-8", errors="ignore").read().split()
    except OSError:
        return []
    words = sorted({w for w in raw if w.isalpha() and w.islower() and 3 <= len(w) <= 10})
    if not words:
        return []
    rng = random.Random(seed)
    out: set[str] = set()
    while len(out) < n:
        k = rng.randint(3, 10)
        ws = [rng.choice(_FUNC) if rng.random() < 0.4 else rng.choice(words) for _ in range(k)]
        out.add((" ".join(ws)).capitalize() + rng.choice([".", ".", "?", ""]))
    return sorted(out)


def split_salad(seed: int = 4321) -> dict[str, list[str]]:
    pool = salad_pool(1400, seed)
    random.Random(seed).shuffle(pool)
    return {"train": pool[:1000], "val": pool[1000:1150], "test": pool[1150:]}


# Short phrases an assistant gets asked after the wake word.
COMMANDS = [
    "what time is it?", "how busy is my CPU right now?", "what's the weather like?", "set a timer for five minutes.",
    "how much memory is free?", "open my downloads folder.", "what's on my calendar today?", "play some music.",
    "is the network up?", "turn the volume down.", "what are the biggest processes?", "how much disk space is left?",
]
