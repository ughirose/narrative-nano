import json
import os
import random

def generate_error_detection_corpus(output_path="dist/datasets/error_benchmark_corpus.jsonl"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    items = []

    # Category 1: Particle Repetition (助詞重複)
    particles_repetition_samples = [
        ("彼が猫が魚が好きだと言った。", "particle-repetition", "が", 3),
        ("勇者が仲間が敵が倒れたと叫んだ。", "particle-repetition", "が", 3),
        ("彼を部屋を荷物を見せに連れて行った。", "particle-repetition", "を", 3),
        ("剣を盾を鎧を落として逃げ出した。", "particle-repetition", "を", 3),
        ("私の友達の犬の家の庭。", "particle-repetition", "の", 4),
        ("学校の図書室の奥の机の上の本。", "particle-repetition", "の", 4),
        ("ギルドのマスターの部屋の鍵の束。", "particle-repetition", "の", 4),
        ("彼に私に先生に報告しに行った。", "particle-repetition", "に", 3),
        ("東京で大阪で名古屋で開催された。", "particle-repetition", "で", 3),
    ]
    for text, err_type, target, count in particles_repetition_samples * 20:
        items.append({
            "text": text,
            "has_error": True,
            "error_type": err_type,
            "target": target,
            "count": count,
            "description": f"同一文内で助詞「{target}」が{count}回重複"
        })

    # Category 2: Subject-Predicate Mismatch (主述のねじれ)
    subject_predicate_mismatch_samples = [
        ("私の将来の夢は、世界大会で優勝したからです。", "subject-predicate-mismatch", "私の夢は〜からです"),
        ("私が最も驚いた理由は、彼が突然走り出したからです。", "clean", "正当な理由構文"), # clean control
        ("彼は、明日晴れると良いなと思ったからです。", "subject-predicate-mismatch", "彼は〜と思ったからです"),
        ("僕の希望としては、全員が無事に帰還できるからです。", "subject-predicate-mismatch", "希望としては〜からです"),
        ("この計画の目的は、敵の拠点を偵察することです。", "clean", "正当な目的構文"), # clean control
        ("彼女の提案は、一旦退却して体制を立て直すからです。", "subject-predicate-mismatch", "提案は〜からです"),
    ]
    for text, err_type, desc in subject_predicate_mismatch_samples * 25:
        items.append({
            "text": text,
            "has_error": err_type != "clean",
            "error_type": err_type,
            "description": desc
        })

    # Category 3: Double Negation & Excessive Passive (二重否定・過剰受身)
    stylistic_error_samples = [
        ("その真実を知らないわけではないと言わざるを得ない。", "double-negation", "二重否定の重畳"),
        ("彼の主張に賛同できなくもないわけではない。", "double-negation", "二重否定の多用"),
        ("敵に城を奪われて、味方が皆殺害された。", "consecutive-passive", "受身の連続"),
        ("魔王に追われて、仲間に見捨てられて、孤立させられた。", "consecutive-passive", "3連続受身"),
        ("静かに扉を開けて、誰もいない部屋に入った。", "clean", "自然な複文"),
        ("主人公は剣を抜き、敵の攻撃を冷静にかわした。", "clean", "自然な能動態"),
    ]
    for text, err_type, desc in stylistic_error_samples * 25:
        items.append({
            "text": text,
            "has_error": err_type != "clean",
            "error_type": err_type,
            "description": desc
        })

    # Category 4: POV Contradiction & Style Shift (視点ブレ・口調崩れ)
    pov_style_samples = [
        ("エリスは恐怖に震えていた。しかし俺は冷たく見下ろした。", "pov-shift", "三人称地の文への急な一人称混入"),
        ("彼は冷徹な暗殺者だった。でも本当はちょっぴり寂しがり屋なのだ。", "style-shift", "ハードボイルド調から甘口文体への急変"),
        ("ギルドの受付嬢は微笑んだ。私は書類を受け取って酒場へ向かった。", "clean", "一貫した一人称視点"),
        ("夕暮れの街を一人歩きながら、過ぎ去った日々に思いを馳せる。", "clean", "一貫した叙情文"),
    ]
    for text, err_type, desc in pov_style_samples * 30:
        items.append({
            "text": text,
            "has_error": err_type != "clean",
            "error_type": err_type,
            "description": desc
        })

    random.seed(42)
    random.shuffle(items)

    with open(output_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    total = len(items)
    err_count = sum(1 for x in items if x["has_error"])
    clean_count = total - err_count

    stats = {
        "total_samples": total,
        "error_samples": err_count,
        "clean_samples": clean_count,
        "error_ratio": round(err_count / total, 4),
        "categories": {
            "particle-repetition": sum(1 for x in items if x.get("error_type") == "particle-repetition"),
            "subject-predicate-mismatch": sum(1 for x in items if x.get("error_type") == "subject-predicate-mismatch"),
            "double-negation": sum(1 for x in items if x.get("error_type") == "double-negation"),
            "consecutive-passive": sum(1 for x in items if x.get("error_type") == "consecutive-passive"),
            "pov-shift": sum(1 for x in items if x.get("error_type") == "pov-shift"),
            "style-shift": sum(1 for x in items if x.get("error_type") == "style-shift"),
            "clean": clean_count,
        }
    }

    stats_path = output_path.replace(".jsonl", "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[SUCCESS] Generated {total} samples -> {output_path}")
    print(f"Stats: {json.dumps(stats, ensure_ascii=False, indent=2)}")
    return stats

if __name__ == "__main__":
    generate_error_detection_corpus()
