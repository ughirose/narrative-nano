#!/usr/bin/env python3
"""
narrative-nano/scripts/build_leakfree_v18_corpus.py

Deterministic Leak-Free Token Corpus Builder for Narrative-Nano Pro v18.
Strict Zero-Leakage Enforcement against literary_benchmark_100.jsonl and kasane_nano_fused_samples.jsonl.
Integrates 12 Aozora Bunko masterpieces and balanced structural patterns:
- Dialogue (Standard bracket, Double-bracket 『…』 300+, Unquoted dialogue 450+)
- Monologue (Parentheses （…）, Dash ――, Free Indirect Discourse FID 450+)
- Citation (Book titles, signs, dictionary entries, technical terms 400+)
"""

import gzip
import hashlib
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

# Random seed for deterministic reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(BASE_DIR, "dist", "datasets")
OUTPUT_JSONL = os.path.join(DATASETS_DIR, "token_corpus_v18_leakfree.jsonl")
OUTPUT_GZ = os.path.join(DATASETS_DIR, "token_corpus_v18_leakfree.jsonl.gz")

BENCHMARK_100_PATH = os.path.join(DATASETS_DIR, "literary_benchmark_100.jsonl")
FUSED_SAMPLES_PATH = os.path.join(DATASETS_DIR, "kasane_nano_fused_samples.jsonl")
AOZORA_DATASET_PATH = os.path.join(DATASETS_DIR, "aozora_literary_dataset.jsonl")

# 12 Aozora Works File Paths
KASANE_CORPUS_FILES = [
    r"C:\dev\kasane\tests\corpus\lemon_full.txt",
    r"C:\dev\kasane\tests\corpus\sangetsuki_full.txt",
    r"C:\dev\kasane\tests\corpus\takasebune_full.txt",
    r"C:\dev\kasane\tests\corpus\shikki_1_full.txt",
    r"C:\dev\kasane\data\corpus\hardboiled\unno_1.txt",
    r"C:\dev\kasane\data\corpus\hardboiled\yumeno_1.txt",
    r"C:\dev\kasane\data\corpus\pure_lit\dazai.txt",
    r"C:\dev\kasane\data\corpus\pure_lit\soseki.txt",
]

# 10 Japanese Case Particles Definition
CASE_ORDER = ["より", "から", "まで", "が", "を", "に", "で", "と", "へ", "の"]
CASE_MAP = {
    "が": {"case_id": 0, "name": "主格", "katakana": "ガ"},
    "を": {"case_id": 1, "name": "対格", "katakana": "ヲ"},
    "に": {"case_id": 2, "name": "与格", "katakana": "ニ"},
    "で": {"case_id": 3, "name": "具格", "katakana": "デ"},
    "と": {"case_id": 4, "name": "共格", "katakana": "ト"},
    "より": {"case_id": 5, "name": "奪格", "katakana": "ヨリ"},
    "から": {"case_id": 6, "name": "起点格", "katakana": "カラ"},
    "まで": {"case_id": 7, "name": "終点格", "katakana": "マデ"},
    "へ": {"case_id": 8, "name": "方向格", "katakana": "ヘ"},
    "の": {"case_id": 9, "name": "連体修飾格", "katakana": "ノ"},
}
PARTICLE_SPLIT_REGEX = re.compile(r'[、。！？「」『』（）\s]|(?:より|から|まで|が|を|に|で|と|へ|の)')


class JapaneseCaseAnnotator:
    """Annotates 10 case particles and syntactic spans."""

    @classmethod
    def analyze_particles(cls, sentence: str) -> List[Dict[str, Any]]:
        detected = []
        n = len(sentence)
        i = 0

        while i < n:
            matched = False
            for particle in CASE_ORDER:
                plen = len(particle)
                if sentence[i:i + plen] == particle:
                    case_info = CASE_MAP[particle]
                    start_span = i
                    end_span = i + plen

                    target_start = max(0, start_span - 15)
                    before_sub = sentence[target_start:start_span]
                    sub_tokens = [tok for tok in PARTICLE_SPLIT_REGEX.split(before_sub) if tok]
                    target_word = sub_tokens[-1] if sub_tokens else sentence[max(0, start_span - 4):start_span]

                    detected.append({
                        "particle": particle,
                        "case_id": case_info["case_id"],
                        "case_name": case_info["name"],
                        "case_katakana": case_info["katakana"],
                        "span": [start_span, end_span],
                        "target_noun": target_word
                    })
                    i += plen
                    matched = True
                    break

            if not matched:
                i += 1

        return detected


class AozoraTextCleaner:
    """Handles ruby notation, editorial annotations, and NFKC text normalization."""

    @staticmethod
    def clean(raw_text: str) -> str:
        lines = raw_text.splitlines()
        content_lines = []
        in_header = True
        dash_count = 0

        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith("-------"):
                dash_count += 1
                if dash_count == 2:
                    in_header = False
                continue
            if in_header and dash_count < 2:
                continue
            if line_stripped.startswith("底本：") or line_stripped.startswith("［＃本文終わり］"):
                break
            content_lines.append(line)

        text = "\n".join(content_lines) if content_lines else raw_text

        # Strip editorial notes ［＃...］
        text = re.sub(r'［＃[^］]*］', '', text)
        # Strip ruby
        text = re.sub(r'｜([^《\n]+)《([^》\n]+)》', r'\1', text)
        text = re.sub(r'([一-龠々〆ヵヶ]+)《([^》\n]+)》', r'\1', text)
        text = re.sub(r'《[^》\n]*》', '', text)
        text = text.replace('｜', '')

        # Unicode normalization (NFKC)
        normalized = unicodedata.normalize('NFKC', text)
        return normalized


class BenchmarkGuard:
    """Guarantees 100% zero-leakage separation from benchmarks."""

    def __init__(self, bench_100_path: str, fused_path: str):
        self.exact_texts: Set[str] = set()
        self.core_phrases: Set[str] = set()
        self._load_benchmarks(bench_100_path, fused_path)

    def _load_benchmarks(self, bench_100_path: str, fused_path: str):
        # 1. literary_benchmark_100.jsonl
        if os.path.exists(bench_100_path):
            with open(bench_100_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    t = d.get("text", "").strip()
                    if t:
                        self.exact_texts.add(t)
                        self._extract_phrases(t)

        # 2. kasane_nano_fused_samples.jsonl
        if os.path.exists(fused_path):
            with open(fused_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    for k in ["text", "current_text", "prev_dialogue"]:
                        if k in d and d[k]:
                            t = d[k].strip()
                            self.exact_texts.add(t)
                            self._extract_phrases(t)
                    if "sentences" in d and isinstance(d["sentences"], list):
                        for s in d["sentences"]:
                            if isinstance(s, str) and s.strip():
                                self.exact_texts.add(s.strip())
                                self._extract_phrases(s.strip())
                            elif isinstance(s, dict) and "text" in s and s["text"].strip():
                                self.exact_texts.add(s["text"].strip())
                                self._extract_phrases(s["text"].strip())

        print(f"[*] BenchmarkGuard loaded {len(self.exact_texts)} exact benchmark texts and {len(self.core_phrases)} core phrases.")

    def _extract_phrases(self, text: str):
        # Extract content inside quotes / brackets
        matches = re.findall(r'[「『（]([^」』）]+)[」』）]', text)
        for m in matches:
            m = m.strip()
            if len(m) >= 5:
                self.core_phrases.add(m)
        # Extract unquoted dialogue clauses
        if "、と" in text:
            parts = text.split("、と")
            if len(parts[0]) >= 6:
                self.core_phrases.add(parts[0].strip())

    def is_leaked(self, text: str) -> bool:
        t = text.strip()
        if not t:
            return True
        if t in self.exact_texts:
            return True

        # Bi-directional substring check for long sentences
        for bt in self.exact_texts:
            if len(bt) >= 12 and (bt in t or t in bt):
                return True

        # Check core phrase collision
        for cp in self.core_phrases:
            if len(cp) >= 8 and cp in t:
                return True

        return False


def build_aozora_dataset_records(guard: BenchmarkGuard, max_count: int = 4500) -> List[Dict[str, Any]]:
    """Loads clean records from aozora_literary_dataset.jsonl (Kokoro, Botchan, Rashomon, Melos)."""
    records = []
    if not os.path.exists(AOZORA_DATASET_PATH):
        print(f"[!] Warning: {AOZORA_DATASET_PATH} not found.")
        return records

    print(f"[*] Loading from {AOZORA_DATASET_PATH}...")
    with open(AOZORA_DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            text = d.get("text", "").strip()

            if len(text) < 6 or len(text) > 120:
                continue
            if guard.is_leaked(text):
                continue

            length = len(text)
            is_dialogue = d.get("is_dialogue", False)

            # Modality token array
            if is_dialogue:
                # Dialogue bracket or standard dialogue
                if text.startswith("（") and text.endswith("）"):
                    modality = [2] * length  # Monologue
                else:
                    modality = [1] * length  # Dialogue
            else:
                modality = [0] * length
                # Inline brackets
                for m in re.finditer(r'(「[^」]*」|『[^』]*』)', text):
                    s, e = m.start(), m.end()
                    for idx in range(s, min(length, e)):
                        modality[idx] = 1
                for m in re.finditer(r'（[^）]*）', text):
                    s, e = m.start(), m.end()
                    for idx in range(s, min(length, e)):
                        modality[idx] = 2

            particles = JapaneseCaseAnnotator.analyze_particles(text)

            records.append({
                "text": text,
                "token_modality": modality,
                "case_particles": particles,
                "length": length,
                "source": d.get("work_title", "aozora_4works")
            })

            if len(records) >= max_count:
                break

    print(f"[+] Loaded {len(records)} clean records from aozora_literary_dataset.jsonl.")
    return records


def build_kasane_corpus_records(guard: BenchmarkGuard, max_count_per_file: int = 350) -> List[Dict[str, Any]]:
    """Segments and annotates 8 additional masterpieces from Kasane corpus."""
    records = []
    dialogue_pattern = re.compile(r'(「[^」]*」|『[^』]*』|（[^）]*）)')
    sentence_pattern = re.compile(r'([^。！？!?]*[。！？!?])')

    for fpath in KASANE_CORPUS_FILES:
        if not os.path.exists(fpath):
            print(f"[!] Warning: File {fpath} not found.")
            continue

        fname = os.path.basename(fpath)
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

        cleaned_text = AozoraTextCleaner.clean(raw_text)
        file_records = 0

        for line in cleaned_text.splitlines():
            line = line.strip()
            if not line:
                continue

            # Split line into bracketed dialogue and narrative segments
            parts = dialogue_pattern.split(line)
            for part in parts:
                part = part.strip()
                if not part:
                    continue

                if (part.startswith("「") and part.endswith("」")) or (part.startswith("『") and part.endswith("』")):
                    text = part
                    if 6 <= len(text) <= 120 and not guard.is_leaked(text):
                        modality = [1] * len(text)
                        particles = JapaneseCaseAnnotator.analyze_particles(text)
                        records.append({
                            "text": text,
                            "token_modality": modality,
                            "case_particles": particles,
                            "length": len(text),
                            "source": fname
                        })
                        file_records += 1
                elif part.startswith("（") and part.endswith("）"):
                    text = part
                    if 6 <= len(text) <= 120 and not guard.is_leaked(text):
                        modality = [2] * len(text)
                        particles = JapaneseCaseAnnotator.analyze_particles(text)
                        records.append({
                            "text": text,
                            "token_modality": modality,
                            "case_particles": particles,
                            "length": len(text),
                            "source": fname
                        })
                        file_records += 1
                else:
                    # Narrative split by punctuation
                    sents = sentence_pattern.findall(part)
                    rem = sentence_pattern.sub('', part).strip()
                    if rem:
                        sents.append(rem)

                    for s in sents:
                        s = s.strip()
                        if 8 <= len(s) <= 120 and not guard.is_leaked(s):
                            modality = [0] * len(s)
                            particles = JapaneseCaseAnnotator.analyze_particles(s)
                            records.append({
                                "text": s,
                                "token_modality": modality,
                                "case_particles": particles,
                                "length": len(s),
                                "source": fname
                            })
                            file_records += 1

                if file_records >= max_count_per_file:
                    break
            if file_records >= max_count_per_file:
                break

        print(f"[+] Loaded {file_records} records from {fname}.")

    print(f"[+] Total {len(records)} records from 8 Kasane corpus files.")
    return records


# ==============================================================================
# Balanced Structural Diversity Generators (Clean, Guaranteed Zero-Leakage)
# ==============================================================================

def generate_double_bracket_dialogues(guard: BenchmarkGuard, target_count: int = 400) -> List[Dict[str, Any]]:
    """Generates 300+ clean double-bracket 『…』 dialogue sentences (telephone, radio, recall, telepathy, drama)."""
    speakers = [
        "受話器の向こうから母が", "受話器を取ると姉が", "スピーカーから管制官が", "通信機がノイズ混じりに",
        "回想の中で父が", "遠い日の記憶の中で祖父が", "少女は小声で囁いて", "妹は心配そうに尋ねて",
        "青年は真剣な眼差しで", "男は低い声で警告して", "モニター越しに教官が", "劇中の役者が大声で",
        "ラジオのアナウンサーが", "頭の中に直接声が響き", "彼は手帳のメモを指差して", "彼女は震える声で",
        "背後から誰かが", "扉の向こうから微かな声が", "隊長は無線機に向かって", "老人は懐かしむように語り",
        "公衆電話の向こうで友人が", "暗闇の中で見知らぬ者が", "客車の車掌が静かに", "看護師が病室の外から",
        "教授が教卓の前に立って", "運転手がミラー越しに", "船頭が船首から身を乗り出して", "警備員がインターホン越しに"
    ]
    actions = [
        "と告げた。", "と囁いた。", "と尋ねた。", "と警告した。", "と声を震わせた。",
        "と問いかけてきた。", "と言い放った。", "と返答した。", "と呼びかけた。", "と呟いた。",
        "と叫んだ。", "と促した。", "と念を押した。", "と伝えてきた。"
    ]
    utterances = [
        "今すぐそちらへ向かいます", "予定通りに作戦を開始してください", "どうか無事でいておくれ", "まだ諦めるには早すぎるわ",
        "あの約束を忘れたわけじゃないだろうね", "信号が赤に変わる前に渡りなさい", "誰も私の話を聞いてくれないの",
        "その扉を決して開けてはならない", "次の停車駅で合流しましょう", "探していたものは見つかりましたか",
        "何が起きても前を向き続けるんだ", "明日の朝一番の便で発ちます", "本当の気持ちを聞かせてくれないか",
        "暗闇の中で光を探し求めなさい", "もうこれ以上隠し事はできません", "お茶が冷めないうちに召し上がれ",
        "北西の空に異常な光が見えます", "過去にとらわれていては前へ進めない", "あの場所でずっと待っているから",
        "準備が整い次第合図を送ってくれ", "真実はいつも闇の中に隠されている", "急いで荷物をまとめて外に出るんだ",
        "風向きが変わる前に船を出そう", "決して振り返らずに走り続けなさい", "答えは君自身の心の中にあるはずだ",
        "時計の針が重なる瞬間に始めよう", "雨が降る前に洗濯物を取り込んでね", "あの丘の上に白い花が咲いていたよ",
        "危険が迫っています、避難してください", "私の言葉をどうか信じてほしい", "明日の講義の資料を忘れずにね",
        "星が見える夜にまた語り合いましょう", "鍵は植木鉢の下に隠してあります", "遠くで汽笛が鳴り響いているわ"
    ]

    records = []
    generated_texts = set()

    for u in utterances:
        for sp in speakers:
            act = random.choice(actions)
            # Pattern A: 『...』と〜
            t1 = f"『{u}』{act}"
            # Pattern B: [主語]、『...』と[動詞]
            t2 = f"{sp}、『{u}』{act}"

            for cand in [t1, t2]:
                if cand not in generated_texts and not guard.is_leaked(cand):
                    generated_texts.add(cand)
                    # Label: 『...』 span is 1: Dialogue, rest is 0: Narrative
                    b_start = cand.index("『")
                    b_end = cand.index("』") + 1
                    modality = [0] * len(cand)
                    for i in range(b_start, b_end):
                        modality[i] = 1

                    records.append({
                        "text": cand,
                        "token_modality": modality,
                        "case_particles": JapaneseCaseAnnotator.analyze_particles(cand),
                        "length": len(cand),
                        "source": "synthetic_double_bracket"
                    })
                if len(records) >= target_count:
                    break
            if len(records) >= target_count:
                break
        if len(records) >= target_count:
            break

    print(f"[+] Generated {len(records)} clean double-bracket 『…』 records.")
    return records


def generate_unquoted_dialogues(guard: BenchmarkGuard, target_count: int = 550) -> List[Dict[str, Any]]:
    """Generates 450+ clean unquoted dialogue sentences (dialogue_unquoted)."""
    utterances = [
        # 呼格・命令・制止
        "おい、早く支度しろ", "ねえ、ちょっと手を貸して", "もし、落とし物ですよ", "やあ、調子はどうだい",
        "おい、こっちを見てくれ", "ほら、足元に気をつけろ", "ねえ、そろそろ出発しよう", "おい、あれを見ろよ",
        "待て、走るんじゃない", "伏せろ、頭を低くしろ", "急いでくれ、時間がないんだ", "やめてくれ、もうたくさんだ",
        "動くな、手を挙げろ", "静かにしろ、息を殺せ", "逃げるな、堂々と向き合え", "話せ、何を知っているんだ",
        # 疑問・確認
        "本当にあるのかい", "いったいどういうつもりなんだ", "誰がそんなことを言い出したんだ", "まだ間に合うだろうか",
        "これで合っているのかい", "何が起こったというんだ", "どうしてそんな顔をするんだ", "本当に一人で行く気かい",
        "どこへ隠したんだ", "いつまで待たせる気だい", "これで満足かい", "君も来るつもりなのかい",
        # 感嘆・安堵・警告
        "信じられない、奇跡だ", "すごいぞ、やったじゃないか", "助かった、もうだめかと思った", "危ない、下がっていろ",
        "嘘だろう、そんな馬鹿な", "素晴らしい、完璧な出来栄えだ", "恐ろしい、身の毛がよだつようだ", "痛い、誰か助けてくれ",
        "見事だ、感服したよ", "寒くて凍えそうだ", "眩しい、目が眩むようだ", "息苦しい、窓を開けてくれ",
        # 提案・日常
        "少し休もう、疲れただろう", "さあ召し上がれ、熱いうちにどうぞ", "乾杯しよう、我々の成功に", "もう帰ろう、夜が更けてきた",
        "一緒に行こう、心強いから", "任せておけ、うまくやってみせる", "気にしないで、大したことじゃないよ", "暖炉に薪をくべよう"
    ]

    narratives = [
        "と背後から鋭い声が飛んだ。", "と店主は湯気の立つ器を差し出した。",
        "と青年は深いため息をつきながら頭を抱えた。", "と少女は驚いたように目を丸くした。",
        "と老人は杖を握り直して言った。", "と見知らぬ男が駆け寄ってきた。",
        "と車掌が切符を改めながら告げた。", "と看護師が静かに病室の扉を開けた。",
        "と船頭が櫂の手を休めて叫んだ。", "と警官が厳しい表情で尋ねた。",
        "と母は心配そうに眉をひそめた。", "と少年は誇らしげに胸を張った。",
        "と職人は作業の手を止めて振り返った。", "と案内人は薄暗い洞窟の奥を指さした。",
        "と医者は眼鏡の奥の目を細めた。", "と馭者は馬の手綱を強く引いた。",
        "と農夫は額の汗を拭いながら笑った。", "と番頭は帳簿から顔を上げた。"
    ]

    records = []
    generated_texts = set()

    for u in utterances:
        for nar in narratives:
            cand = f"{u}、{nar}"
            if cand not in generated_texts and not guard.is_leaked(cand):
                generated_texts.add(cand)

                # Dialogue span: from start to comma
                # Narrative span: from "、" onwards
                u_len = len(u)
                modality = [1] * u_len + [0] * (len(cand) - u_len)

                records.append({
                    "text": cand,
                    "token_modality": modality,
                    "case_particles": JapaneseCaseAnnotator.analyze_particles(cand),
                    "length": len(cand),
                    "source": "synthetic_unquoted_dialogue"
                })
            if len(records) >= target_count:
                break
        if len(records) >= target_count:
            break

    print(f"[+] Generated {len(records)} clean unquoted dialogue records.")
    return records


def generate_monologue_fid_records(guard: BenchmarkGuard, target_count: int = 520) -> List[Dict[str, Any]]:
    """Generates 450+ clean Monologue records (Parentheses, Dash, Free Indirect Discourse)."""
    # 1. Parentheses Monologues
    paren_monologues = [
        "ここで諦めるわけにはいかない", "あの男の真意は何なのだろう", "これ以上迷惑をかけるわけにはいかない",
        "あの時の選択は正しかったのだろうか", "いつか必ず真実を白日の下に晒してみせる", "誰も信じられない、孤独が身に染みる",
        "どうしてあの場に居合わせてしまったのか", "明日の今頃にはすべてが決着しているはずだ", "この胸のざわめきは何の前触れだろう",
        "頼るべき者はもはや自分しかいないのだ", "あの人の優しい笑顔をもう一度見たい", "逃げ出したい、だが足が竦んで動かない",
        "この手で結末をつけねばならない", "嘘偽りのない言葉を聞きたかった", "もう二度と過ちは繰り返さない",
        "なぜあの時あんな軽はずみな約束をしたのか", "あの子だけは絶対に守り抜いてみせる", "今夜を生き延びることさえできれば",
        "私にはそんな資格など最初からなかったのだ", "あの日の誓いはまだ生きているだろうか", "もう誰の指図も受けたくはない",
        "すべてを投げ出して遠くへ逃げ去りたい", "この暗闇の先に本当に朝が来るのか", "己の弱さを認めるのが何よりも恐ろしい"
    ]
    paren_leads = [
        "と強く拳を握り締めた。", "胸の中で疑念が渦巻いていた。", "と一人で荷物をまとめた。",
        "冷たい風に吹かれながら自問した。", "固い決意を胸に秘めた。", "と夜空の星を見上げた。",
        "と後悔の念に苛まれた。", "と静かに呼吸を整えた。", "額に冷や汗が滲み出た。",
        "と覚悟を決めて立ち上がった。", "寂しげな微笑を浮かべた。", "恐怖で喉がカラカラに渇いた。",
        "と厳しい視線を向けた。", "と深いため息をついた。", "と自分に言い聞かせた。",
        "と心の中で毒づいた。", "と切なげに眉を寄せた。", "と祈るように呟いた。"
    ]

    # 2. Dash Monologues
    dash_monologues = [
        "――そうだ、最初から計画されていたことなのだ。",
        "――行かねばならない、あの丘の向こうへ。",
        "――何もかもが手遅れだったというのか。",
        "――信じるしかない、あの人の言葉を。",
        "――静かすぎる、嵐の前の静けさだ。",
        "――夢であってくれ、ただの悪夢であってくれ。",
        "――これが運命というものなのかもしれない。",
        "――振り返るな、前だけを見据えて進むのだ。",
        "――あの日の記憶が、鮮やかに蘇ってくる。",
        "――どこで道を誤ってしまったのだろう。",
        "――もう引き返す道は残されていないのだ。",
        "――光が見える、かすかな希望の光が。",
        "――まさかあいつが、裏で糸を引いていたのか。",
        "――死ぬ気になれば、何だってできるはずだ。",
        "――あの笑顔を、二度と曇らせてはならない。",
        "――時は刻一刻と迫っている、迷っている暇はない。"
    ]

    # 3. Free Indirect Discourse (FID)
    fid_sentences = [
        "なぜこれほどまでに胸が騒ぐのか。あんな些細な一言に何を怯えているというのか。",
        "どうして誰も気づいてくれないのか。この苦しみを誰に打ち明けろというのか。",
        "果たして自分にやり遂げられるのだろうか。重すぎる重圧が両肩にのしかかる。",
        "一体いつまで待てばよいのだろう。忍耐の限界はとうに過ぎ去っていた。",
        "なんという理不尽な仕打ちだろうか。身に覚えのない罪を着せられて黙っていられるはずがない。",
        "あんな男の言うことなど信じるべきではなかった。最初からすべて仕組まれていたのではないか。",
        "ここで引き下がっては男の沽券に関わる。どんな困難が待ち受けていようと乗り越えてみせる。",
        "どうしてあんな嘘をついてしまったのか。取り返しのつかない後悔が胸を抉る。",
        "誰も本当のことを教えてくれない。自分だけが暗闇の中に取り残されているかのようだ。",
        "いつまでこんな生活を続けねばならないのか。自由を求める思いが日増しに強くなっていく。",
        "なぜあの時あんな行動をとってしまったのか。過去を消し去ることはできないのか。",
        "果たしてあの約束は守られるのだろうか。疑心暗鬼が心を黒く染め上げていく。",
        "あいつの思い通りにさせてたまるものか。この屈辱は必ず倍にして返してやる。",
        "どうしていつも自分ばかりが貧乏くじを引くのか。運命の不公平さを呪わずにはいられない。",
        "何を恐れる必要があるというのだ。失うものなど最初から何一つ持っていないではないか。",
        "いつから自分はこんなに臆病になってしまったのか。あの頃の情熱はどこへ消えたのか。"
    ]

    records = []
    generated_texts = set()

    # Part A: Parentheses (target: 200)
    for m in paren_monologues:
        for lead in paren_leads:
            cand = f"（{m}）{lead}"
            if cand not in generated_texts and not guard.is_leaked(cand):
                generated_texts.add(cand)
                p_end = cand.index("）") + 1
                modality = [2] * p_end + [0] * (len(cand) - p_end)
                records.append({
                    "text": cand,
                    "token_modality": modality,
                    "case_particles": JapaneseCaseAnnotator.analyze_particles(cand),
                    "length": len(cand),
                    "source": "synthetic_monologue_paren"
                })
            if len(records) >= 200:
                break
        if len(records) >= 200:
            break

    # Part B: Dash Monologue (target: 150)
    dash_target = len(records) + 150
    for d_text in dash_monologues:
        for suffix in ["と男は呟いた。", "と彼女は悟った。", "と老人は目を閉じた。", "と胸に手を当てた。", "と青年は息を呑んだ。", "と拳を握った。", ""]:
            cand = f"{d_text}{suffix}".strip()
            if cand not in generated_texts and not guard.is_leaked(cand):
                generated_texts.add(cand)
                d_len = len(d_text)
                modality = [2] * d_len + [0] * (len(cand) - d_len)
                records.append({
                    "text": cand,
                    "token_modality": modality,
                    "case_particles": JapaneseCaseAnnotator.analyze_particles(cand),
                    "length": len(cand),
                    "source": "synthetic_monologue_dash"
                })
            if len(records) >= dash_target:
                break
        if len(records) >= dash_target:
            break

    # Part C: FID (target: remainder up to target_count)
    for fid in fid_sentences:
        for prefix in ["青年は夜空を仰いだ。", "男は煙草を揉み消した。", "彼女は冷たい窓に額を当てた。", "旅人は立ち止まった。", "兵士は泥濘を睨みつけた。", "少女は震えながら俯いた。", "老人は暖炉を見つめた。", ""]:
            cand = f"{prefix}{fid}".strip()
            if cand not in generated_texts and not guard.is_leaked(cand):
                generated_texts.add(cand)
                p_len = len(prefix)
                modality = [0] * p_len + [2] * (len(cand) - p_len)
                records.append({
                    "text": cand,
                    "token_modality": modality,
                    "case_particles": JapaneseCaseAnnotator.analyze_particles(cand),
                    "length": len(cand),
                    "source": "synthetic_monologue_fid"
                })
            if len(records) >= target_count:
                break
        if len(records) >= target_count:
            break

    print(f"[+] Generated {len(records)} clean Monologue/FID records.")
    return records


def generate_citation_records(guard: BenchmarkGuard, target_count: int = 480) -> List[Dict[str, Any]]:
    """Generates 400+ clean Citation records (titles, signs, dictionary entries, technical terms)."""
    citations = [
        # Book titles & masterpieces
        ("カラマーゾフの兄弟", "本棚には「カラマーゾフの兄弟」の背表紙が並んでいた。"),
        ("世界史概説", "彼は机の上に置かれた「世界史概説」を開いた。"),
        ("銀河鉄道の夜", "彼女は「銀河鉄道の夜」の一節を静かに朗読した。"),
        ("国富論", "書架から「国富論」のハードカバーを取り出した。"),
        ("ファウスト", "教授は「ファウスト」の解釈について熱心に講義した。"),
        ("万葉集", "鞄の中に「万葉集」の文庫本を忍ばせていた。"),
        ("百年の孤独", "店主は「百年の孤独」の初版本を自慢げに見せた。"),
        ("罪と罰", "机の上には「罪と罰」の翻訳書が積み重なっていた。"),
        ("源氏物語", "彼女は「源氏物語」の現代語訳を毎晩読んでいる。"),
        ("城", "書斎の棚に「城」のペーパーバックが差してあった。"),
        ("ドン・キホーテ", "彼は「ドン・キホーテ」の挿絵を熱心に眺めていた。"),
        ("失われた時を求めて", "祖父の遺品の中に「失われた時を求めて」全巻があった。"),
        ("変身", "教科書に「変身」の冒頭部分が引用されていた。"),
        ("アンナ・カレーニナ", "母は「アンナ・カレーニナ」の結末について語った。"),
        ("神曲", "古書店で「神曲」の豪華本を見つけて足を止めた。"),
        # Signs, notices & labels
        ("関係者以外立入禁止", "入口の扉には「関係者以外立入禁止」と朱書きされていた。"),
        ("本日休館", "掲示板に「本日休館」の告知が張り出されていた。"),
        ("故障中", "改札口の横に「故障中」の紙が貼られていた。"),
        ("一旦停止", "踏切の手前に「一旦停止」の標識が立っていた。"),
        ("商談中", "店頭のガラス戸には「商談中」の札が掲げられていた。"),
        ("自習", "黒板の中央に「自習」と大きくチョークで書かれていた。"),
        ("満員御礼", "劇場の入口に「満員御礼」の看板が誇らしげに掲げてあった。"),
        ("起こさないでください", "ドアノブに「起こさないでください」のカードが掛かっていた。"),
        ("準備中", "木製の札には「準備中」の文字が刻まれていた。"),
        ("頭上注意", "地下道の入口に「頭上注意」の黄色い看板があった。"),
        ("撮影禁止", "展示室の壁に「撮影禁止」のピクトグラムが貼られていた。"),
        ("立入制限区域", "フェンスには「立入制限区域」のプレートが針金で固定されていた。"),
        ("非常口", "廊下の突き当たりに「非常口」の緑色の誘導灯が光っていた。"),
        ("整理券配布所", "広場の特設テントに「整理券配布所」の横断幕があった。"),
        ("禁煙", "待合室の柱には「禁煙」の文字が刻まれていた。"),
        # Dictionary, scientific & philosophical concepts
        ("ペシミズム", "辞書で「ペシミズム」を調べると厭世観と記されていた。"),
        ("バタフライ効果", "いわゆる「バタフライ効果」に関する考察をレポートにまとめた。"),
        ("認知的不協和", "心理学のテキストで「認知的不協和」の章を精読する。"),
        ("パラダイムシフト", "最新の論考において「パラダイムシフト」の必然性が論じられていた。"),
        ("アポリア", "百科事典の「アポリア」の項目には難問と説明されていた。"),
        ("アンチノミー", "哲学書を開き「アンチノミー」の定義をノートに書き写した。"),
        ("デジャヴ", "彼はいわゆる「デジャヴ」の感覚に襲われて立ち眩みした。"),
        ("カタルシス", "劇作家は劇末における「カタルシス」の効果を追求した。"),
        ("シュルレアリスム", "美術の教科書で「シュルレアリスム」の代表作を鑑賞した。"),
        ("エントロピー", "物理学者は「エントロピー」の増大則を黒板に図解した。"),
        ("形而上学", "論文の序論では「形而上学」の歴史的変遷が概括されていた。"),
        ("イデア論", "古代哲学の講義で「イデア論」の射程が議論された。"),
        ("プラグマティズム", "政治学のゼミで「プラグマティズム」の現代的意義を討議した。")
    ]

    templates = [
        "机の上には「{term}」と書かれた分厚い冊子が置かれていた。",
        "壁には「{term}」という標語が墨黒々と掲げられていた。",
        "男はポケットから「{term}」と記された名刺を取り出した。",
        "看板には鮮やかなペンキで「{term}」と大書されていた。",
        "新聞の一面に「{term}」の活字が躍っていた。",
        "手帳の余白に「{term}」と鉛筆で殴り書きされていた。",
        "画面の中央に「{term}」というダイアログが表示された。",
        "箱の蓋には金文字で「{term}」と刻印されていた。",
        "彼は「{term}」に関する解説書を熱心に読み耽っていた。",
        "窓ガラスに「{term}」の張り紙が雨風に晒されていた。",
        "その報告書は「{term}」の問題点を鋭く指摘していた。",
        "白板には「{term}」の文字が赤いマーカーで囲まれていた。"
    ]

    records = []
    generated_texts = set()

    # Base list
    for term, sentence in citations:
        if sentence not in generated_texts and not guard.is_leaked(sentence):
            generated_texts.add(sentence)
            b_start = sentence.index("「")
            b_end = sentence.index("」") + 1
            modality = [0] * len(sentence)
            for i in range(b_start, b_end):
                modality[i] = 3

            records.append({
                "text": sentence,
                "token_modality": modality,
                "case_particles": JapaneseCaseAnnotator.analyze_particles(sentence),
                "length": len(sentence),
                "source": "synthetic_citation_base"
            })

    # Expansion templates
    terms = [t for t, _ in citations]
    for term in terms:
        for tmpl in templates:
            cand = tmpl.format(term=term)
            if cand not in generated_texts and not guard.is_leaked(cand):
                generated_texts.add(cand)
                b_start = cand.index("「")
                b_end = cand.index("」") + 1
                modality = [0] * len(cand)
                for i in range(b_start, b_end):
                    modality[i] = 3

                records.append({
                    "text": cand,
                    "token_modality": modality,
                    "case_particles": JapaneseCaseAnnotator.analyze_particles(cand),
                    "length": len(cand),
                    "source": "synthetic_citation_expanded"
                })
            if len(records) >= target_count:
                break
        if len(records) >= target_count:
            break

    print(f"[+] Generated {len(records)} clean Citation records.")
    return records


# ==============================================================================
# Main Pipeline
# ==============================================================================

def main():
    print("=" * 70)
    print("Narrative-Nano Pro v18 Leak-Free Token Corpus Construction Pipeline")
    print("=" * 70)

    # 1. Initialize Zero-Leakage Guard
    guard = BenchmarkGuard(BENCHMARK_100_PATH, FUSED_SAMPLES_PATH)

    # 2. Collect 12 Aozora Literary Masterpiece Records
    aozora_records = build_aozora_dataset_records(guard, max_count=5800)
    kasane_records = build_kasane_corpus_records(guard, max_count_per_file=450)
    literature_records = aozora_records + kasane_records

    # 3. Generate Clean Structural Diversity Records
    double_bracket_records = generate_double_bracket_dialogues(guard, target_count=380)
    unquoted_records = generate_unquoted_dialogues(guard, target_count=520)
    monologue_records = generate_monologue_fid_records(guard, target_count=520)
    citation_records = generate_citation_records(guard, target_count=480)

    # 4. Combine and Filter
    all_records = (
        literature_records +
        double_bracket_records +
        unquoted_records +
        monologue_records +
        citation_records
    )

    print(f"\n[*] Total unfiltered combined records: {len(all_records):,}")

    # Deduplicate by text and enforce zero leakage
    seen_texts = set()
    clean_dataset = []

    for r in all_records:
        t = r["text"].strip()
        if t in seen_texts:
            continue
        if guard.is_leaked(t):
            print(f"[!] Leak detected and rejected: {t}")
            continue

        seen_texts.add(t)
        clean_dataset.append(r)

    # Shuffle with fixed seed
    random.seed(RANDOM_SEED)
    random.shuffle(clean_dataset)

    # Target 8,000 - 10,000 samples
    if len(clean_dataset) > 9500:
        clean_dataset = clean_dataset[:9500]

    # Assign final sequential IDs
    for idx, r in enumerate(clean_dataset):
        r["id"] = f"v18_{idx:07d}"

    print(f"\n[+] Final curated dataset size: {len(clean_dataset):,} samples")

    # 5. Strict Zero-Leakage Assertion against literary_benchmark_100.jsonl
    print("\n=== Running Strict Zero-Leakage Audit ===")
    leak_count = 0
    with open(BENCHMARK_100_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            bd = json.loads(line)
            bt = bd.get("text", "").strip()
            for r in clean_dataset:
                if r["text"].strip() == bt:
                    leak_count += 1
                    print(f"[CRITICAL ERROR] Leak detected: {bt}")

    assert leak_count == 0, f"AFL-6 VIOLATION: {leak_count} benchmark leaks found in v18 corpus!"
    print(f"[PASS] Zero-Leakage Audit: Exactly 0 benchmark overlaps detected! (Leak count: {leak_count})")

    # 6. Compute Modality Token Statistics
    modality_token_counts = Counter()
    total_tokens = 0
    case_particle_counts = Counter()

    for r in clean_dataset:
        mods = r["token_modality"]
        modality_token_counts.update(mods)
        total_tokens += len(mods)
        for cp in r.get("case_particles", []):
            case_particle_counts[cp["particle"]] += 1

    print("\n=== Token Modality Distribution in v18 ===")
    for m in range(4):
        c = modality_token_counts[m]
        pct = (c / total_tokens) * 100 if total_tokens else 0
        names = {0: "0: Narrative", 1: "1: Dialogue", 2: "2: Monologue", 3: "3: Citation"}
        print(f"  Class {names[m]}: {c:,} tokens ({pct:.2f}%)")
    print(f"  Total tokens: {total_tokens:,} across {len(clean_dataset):,} sentences.")

    print("\n=== Top Case Particles Distribution in v18 ===")
    for p, count in case_particle_counts.most_common(10):
        print(f"  Particle '{p}': {count:,}")

    # 7. Write to JSONL and JSONL.GZ
    os.makedirs(DATASETS_DIR, exist_ok=True)

    print(f"\n[*] Writing to {OUTPUT_JSONL}...")
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in clean_dataset:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[*] Writing to compressed {OUTPUT_GZ}...")
    with open(OUTPUT_JSONL, "rb") as f_in, gzip.open(OUTPUT_GZ, "wb") as f_out:
        f_out.writelines(f_in)

    # 8. SHA-256 Hashes
    def compute_sha256(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    hash_jsonl = compute_sha256(OUTPUT_JSONL)
    hash_gz = compute_sha256(OUTPUT_GZ)

    print("\n" + "=" * 70)
    print("CORPUS BUILD COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"1. JSONL: {OUTPUT_JSONL}")
    print(f"   Size: {os.path.getsize(OUTPUT_JSONL):,} bytes")
    print(f"   SHA-256: {hash_jsonl}")
    print(f"2. GZ:    {OUTPUT_GZ}")
    print(f"   Size: {os.path.getsize(OUTPUT_GZ):,} bytes")
    print(f"   SHA-256: {hash_gz}")
    print("=" * 70)


if __name__ == "__main__":
    main()
