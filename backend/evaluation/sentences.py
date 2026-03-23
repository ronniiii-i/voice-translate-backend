"""
sentences.py — All test content for WER, BLEU, context, and latency evaluations.

IMPORTS USED BY EACH SCRIPT:
    eval_wer.py      → ASR_SENTENCES
    eval_bleu.py     → BLEU_SENTENCES
    eval_context.py  → CONTEXT_SEQUENCES
    eval_latency.py  → (uses recordings from eval_wer.py, no import needed)
"""

# ── ASR_SENTENCES ─────────────────────────────────────────────────────────────
# 10 sentences per language. Record yourself reading each one.
# File names: en_01.wav ... zh_10.wav in evaluation/recordings/
# The strings here are the GROUND TRUTH — must match exactly what you said.

ASR_SENTENCES = {
    "en": [
        "The meeting starts at three o'clock.",
        "Can you send me the report by tomorrow?",
        "I will be working from home this week.",
        "The project deadline has been moved to Friday.",
        "Please make sure to review the document carefully.",
        "We need to schedule a call with the client.",
        "The new system is running much faster now.",
        "I have already submitted my part of the assignment.",
        "Could you help me understand this problem?",
        "The conference room is booked until five.",
    ],
    "fr": [
        "La réunion commence à trois heures.",
        "Pouvez-vous m'envoyer le rapport demain?",
        "Je travaillerai depuis chez moi cette semaine.",
        "La date limite du projet a été reportée à vendredi.",
        "Veuillez vous assurer de bien relire le document.",
        "Nous devons planifier un appel avec le client.",
        "Le nouveau système fonctionne beaucoup plus vite maintenant.",
        "J'ai déjà soumis ma partie du devoir.",
        "Pouvez-vous m'aider à comprendre ce problème?",
        "La salle de conférence est réservée jusqu'à cinq heures.",
    ],
    "de": [
        "Das Meeting beginnt um drei Uhr.",
        "Können Sie mir den Bericht bis morgen schicken?",
        "Ich werde diese Woche von zu Hause arbeiten.",
        "Die Projektfrist wurde auf Freitag verschoben.",
        "Bitte stellen Sie sicher, das Dokument sorgfältig zu prüfen.",
        "Wir müssen einen Anruf mit dem Kunden planen.",
        "Das neue System läuft jetzt viel schneller.",
        "Ich habe meinen Teil der Aufgabe bereits eingereicht.",
        "Könnten Sie mir bei diesem Problem helfen?",
        "Der Konferenzraum ist bis fünf Uhr gebucht.",
    ],
    "es": [
        "La reunión comienza a las tres.",
        "¿Puede enviarme el informe mañana?",
        "Trabajaré desde casa esta semana.",
        "La fecha límite del proyecto se ha movido al viernes.",
        "Por favor asegúrese de revisar el documento con cuidado.",
        "Necesitamos programar una llamada con el cliente.",
        "El nuevo sistema funciona mucho más rápido ahora.",
        "Ya he enviado mi parte de la tarea.",
        "¿Podría ayudarme a entender este problema?",
        "La sala de conferencias está reservada hasta las cinco.",
    ],
    "zh": [
        "会议三点开始。",
        "你能明天把报告发给我吗？",
        "我这周会在家工作。",
        "项目截止日期推迟到周五了。",
        "请确保仔细审查文件。",
        "我们需要安排一个与客户的通话。",
        "新系统现在运行快多了。",
        "我已经提交了我的作业部分。",
        "你能帮我理解这个问题吗？",
        "会议室预订到五点。",
    ],
}


# ── BLEU_SENTENCES ────────────────────────────────────────────────────────────
# 10 sentences per route with reference translations.
# Keyed by (src_lang, tgt_lang) tuple.
#
# IMPORTANT: Verify reference translations with a native speaker or DeepL
# before running eval_bleu.py. Inaccurate references produce unreliable scores.
#
# PIVOT PAIRS (fr/de/es ↔ zh) are included — the MT model handles the
# English pivot automatically, but you still need reference translations
# for the final target language to score against.

_EN = [
    "The meeting starts at three o'clock.",
    "Can you send me the report by tomorrow?",
    "I will be working from home this week.",
    "The project deadline has been moved to Friday.",
    "Please make sure to review the document carefully.",
    "We need to schedule a call with the client.",
    "The new system is running much faster now.",
    "I have already submitted my part of the assignment.",
    "Could you help me understand this problem?",
    "The conference room is booked until five.",
]

_FR = [
    "La réunion commence à 15 heures.",
    "Peux-tu m'envoyer le rapport d'ici demain?",
    "Je travaillerai à domicile cette semaine.",
    "La date limite du projet a été repoussée à vendredi.",
    "Assure-toi de relire attentivement le document.",
    "Nous devons programmer un appel avec le client.",
    "Le nouveau système fonctionne beaucoup plus rapidement maintenant.",
    "J'ai déjà rendu ma partie du devoir.",
    "Pourriez-vous m'aider à comprendre ce problème?",
    "La salle de conférence est réservée jusqu'à cinq heures.",
]

_DE = [
    "Das Meeting beginnt um drei Uhr.",
    "Können Sie mir den Bericht bis morgen schicken?",
    "Ich werde diese Woche von zu Hause arbeiten.",
    "Die Projektfrist wurde auf Freitag verschoben.",
    "Bitte stellen Sie sicher, das Dokument sorgfältig zu prüfen.",
    "Wir müssen einen Anruf mit dem Kunden planen.",
    "Das neue System läuft jetzt viel schneller.",
    "Ich habe meinen Teil der Aufgabe bereits eingereicht.",
    "Könnten Sie mir bei diesem Problem helfen?",
    "Der Konferenzraum ist bis fünf Uhr gebucht.",
]

_ES = [
    "La reunión empieza a las tres",
    "¿Me puedes enviar el informe para mañana?",
    "Esta semana trabajaré desde casa",
    "La fecha límite del proyecto se ha pospuesto hasta el viernes",
    "Por favor, asegúrate de revisar el documento con atención",
    "Necesitamos programar una llamada con el cliente.",
    "El nuevo sistema funciona mucho más rápido ahora.",
    "Ya he enviado mi parte de la tarea.",
    "¿Podría ayudarme a entender este problema?",
    "La sala de conferencias está reservada hasta las cinco.",
]

_ZH = [
    "会议三点开始",
    "你能明天前把报告发给我吗",
    "这周我会在家办公",
    "项目截止日期已推迟到周五",
    "请务必仔细审阅这份文件",
    "我们需要安排一次与客户的电话会议",
    "新系统现在运行得快多了",
    "我已经提交了我负责的部分作业了",
    "你能帮我理解这个问题吗",
    "会议室已经预订到五点了",
]

# FR→DE references (French source, German target)
_FR_DE = [
    "Das Meeting beginnt um drei Uhr.",
    "Können Sie mir den Bericht bis morgen schicken?",
    "Ich werde diese Woche von zu Hause arbeiten.",
    "Die Projektfrist wurde auf Freitag verschoben.",
    "Bitte stellen Sie sicher, das Dokument sorgfältig zu prüfen.",
    "Wir müssen einen Anruf mit dem Kunden planen.",
    "Das neue System läuft jetzt viel schneller.",
    "Ich habe meinen Teil der Aufgabe bereits eingereicht.",
    "Könnten Sie mir bei diesem Problem helfen?",
    "Der Konferenzraum ist bis fünf Uhr gebucht.",
]

# DE→FR references (German source, French target)
_DE_FR = [
    "La réunion commence à 15 heures.",
    "Peux-tu m'envoyer le rapport d'ici demain?",
    "Je travaillerai à domicile cette semaine.",
    "La date limite du projet a été repoussée à vendredi.",
    "Assure-toi de relire attentivement le document.",
    "Nous devons programmer un appel avec le client.",
    "Le nouveau système fonctionne beaucoup plus rapidement maintenant.",
    "J'ai déjà rendu ma partie du devoir.",
    "Pourriez-vous m'aider à comprendre ce problème?",
    "La salle de conférence est réservée jusqu'à cinq heures.",
]

# FR→ES references
_FR_ES = [
    "La reunión empieza a las tres.",
    "¿Me puedes enviar el informe para mañana?",
    "Esta semana trabajaré desde casa.",
    "La fecha límite del proyecto se ha pospuesto hasta el viernes.",
    "Por favor, asegúrate de revisar el documento con atención.",
    "Necesitamos programar una llamada con el cliente.",
    "El nuevo sistema funciona mucho más rápido ahora.",
    "Ya he enviado mi parte de la tarea.",
    "¿Podría ayudarme a entender este problema?",
    "La sala de conferencias está reservada hasta las cinco.",
]

# ES→FR references
_ES_FR = [
    "La réunion commence à 15 heures.",
    "Peux-tu m'envoyer le rapport d'ici demain?",
    "Je travaillerai à domicile cette semaine.",
    "La date limite du projet a été repoussée à vendredi.",
    "Assure-toi de relire attentivement le document.",
    "Nous devons programmer un appel avec le client.",
    "Le nouveau système fonctionne beaucoup plus rapidement maintenant.",
    "J'ai déjà rendu ma partie du devoir.",
    "Pourriez-vous m'aider à comprendre ce problème?",
    "La salle de conférence est réservée jusqu'à cinq heures.",
]

# DE→ES references
_DE_ES = [
    "La reunión empieza a las tres.",
    "¿Me puedes enviar el informe para mañana?",
    "Esta semana trabajaré desde casa.",
    "La fecha límite del proyecto se ha pospuesto hasta el viernes.",
    "Por favor, asegúrate de revisar el documento con atención.",
    "Necesitamos programar una llamada con el cliente.",
    "El nuevo sistema funciona mucho más rápido ahora.",
    "Ya he enviado mi parte de la tarea.",
    "¿Podría ayudarme a entender este problema?",
    "La sala de conferencias está reservada hasta las cinco.",
]

# ES→DE references
_ES_DE = [
    "Das Meeting beginnt um drei Uhr.",
    "Können Sie mir den Bericht bis morgen schicken?",
    "Ich werde diese Woche von zu Hause arbeiten.",
    "Die Projektfrist wurde auf Freitag verschoben.",
    "Bitte stellen Sie sicher, das Dokument sorgfältig zu prüfen.",
    "Wir müssen einen Anruf mit dem Kunden planen.",
    "Das neue System läuft jetzt viel schneller.",
    "Ich habe meinen Teil der Aufgabe bereits eingereicht.",
    "Könnten Sie mir bei diesem Problem helfen?",
    "Der Konferenzraum ist bis fünf Uhr gebucht.",
]

# Pivot pair references — Chinese targets
# These are what the system should ultimately produce after pivoting via English.
# Verify these with a native Chinese speaker or trusted translation tool.
_FR_ZH = _ZH   # French source → Chinese target (same sentences, same meaning)
_ZH_FR = _FR   # Chinese source → French target
_DE_ZH = _ZH
_ZH_DE = _DE
_ES_ZH = _ZH
_ZH_ES = _ES


BLEU_SENTENCES = {
    # ── English ↔ others (direct) ─────────────────────────────────────────────
    ("en", "fr"): {"sources": _EN, "references": _FR},
    ("fr", "en"): {"sources": _FR, "references": _EN},
    ("en", "de"): {"sources": _EN, "references": _DE},
    ("de", "en"): {"sources": _DE, "references": _EN},
    ("en", "es"): {"sources": _EN, "references": _ES},
    ("es", "en"): {"sources": _ES, "references": _EN},
    ("en", "zh"): {"sources": _EN, "references": _ZH},
    ("zh", "en"): {"sources": _ZH, "references": _EN},

    # ── European cross-pairs (direct) ─────────────────────────────────────────
    ("fr", "de"): {"sources": _FR, "references": _FR_DE},
    ("de", "fr"): {"sources": _DE, "references": _DE_FR},
    ("fr", "es"): {"sources": _FR, "references": _FR_ES},
    ("es", "fr"): {"sources": _ES, "references": _ES_FR},
    ("de", "es"): {"sources": _DE, "references": _DE_ES},
    ("es", "de"): {"sources": _ES, "references": _ES_DE},

    # ── Pivot pairs (via English) ─────────────────────────────────────────────
    ("fr", "zh"): {"sources": _FR, "references": _FR_ZH},
    ("zh", "fr"): {"sources": _ZH, "references": _ZH_FR},
    ("de", "zh"): {"sources": _DE, "references": _DE_ZH},
    ("zh", "de"): {"sources": _ZH, "references": _ZH_DE},
    ("es", "zh"): {"sources": _ES, "references": _ES_ZH},
    ("zh", "es"): {"sources": _ZH, "references": _ZH_ES},
}


# ── CONTEXT_SEQUENCES ─────────────────────────────────────────────────────────
# Discourse-dependent sequences for Table 3 (context A/B evaluation).
# Each sequence ends with a turn containing a pronoun or reference that
# requires prior context to resolve correctly.

CONTEXT_SEQUENCES = [
    {
        "context": ["Maria called earlier."],
        "target":  "She said the project is delayed.",
        "note":    "'She' should resolve to Maria",
        "reference_fr": "Maria a dit que le projet est en retard.",
        "reference_de": "Maria sagte, das Projekt verzögert sich.",
        "reference_zh": "玛丽亚说项目延误了。",
    },
    {
        "context": ["I bought a new laptop yesterday."],
        "target":  "It keeps overheating.",
        "note":    "'It' should resolve to the laptop",
        "reference_fr": "L'ordinateur portable continue de surchauffer.",
        "reference_de": "Der Laptop überhitzt immer wieder.",
        "reference_zh": "那台笔记本电脑一直过热。",
    },
    {
    "context": ["David joined the company last month."],
    "target":  "He is leading the new project.",
    "note":    "'He' should resolve to David",
    "reference_fr": "David dirige le nouveau projet.",
    "reference_de": "David leitet das neue Projekt.",
    "reference_zh": "大卫正在领导新项目。",
    },
    {
        "context": ["The client submitted a complaint.", "We need to handle it carefully."],
        "target":  "They are very unhappy with the service.",
        "note":    "'They' should resolve to the client",
        "reference_fr": "Le client est très mécontent du service.",
        "reference_de": "Der Kunde ist sehr unzufrieden mit dem Service.",
        "reference_zh": "那位客户对服务非常不满意。",
    },
    {
        "context": ["James finished the report this morning."],
        "target":  "He submitted it before the deadline.",
        "note":    "'He' = James, 'it' = the report",
        "reference_fr": "James l'a soumis avant la date limite.",
        "reference_de": "James hat es vor der Frist eingereicht.",
        "reference_zh": "詹姆斯在截止日期前提交了报告。",
    },
    {
        "context": ["The server crashed last night."],
        "target":  "It has been restored since then.",
        "note":    "'It' should resolve to the server",
        "reference_fr": "Le serveur a été restauré depuis lors.",
        "reference_de": "Der Server wurde seitdem wiederhergestellt.",
        "reference_zh": "服务器此后已恢复正常。",
    },  
    {
        "context": ["Sarah presented the quarterly results."],
        "target":  "She did an excellent job.",
        "note":    "'She' should resolve to Sarah",
        "reference_fr": "Sarah a fait un excellent travail.",
        "reference_de": "Sarah hat hervorragende Arbeit geleistet.",
        "reference_zh": "莎拉做得非常出色。",
    },
    {
        "context": ["The company released a new product last week."],
        "target":  "It has already received very positive reviews.",
        "note":    "'It' = the product",
        "reference_fr": "Le produit a déjà reçu des avis très positifs.",
        "reference_de": "Das Produkt hat bereits sehr positive Bewertungen erhalten.",
        "reference_zh": "该产品已经收到了非常积极的评价。",
    },
]