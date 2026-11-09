#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_kasane_fused_samples.py
Deterministic dataset generator for Kasane stylistic telemetry & Narrative-Nano integration stress testing.
Generates 120 verified literary prose samples across 6 distinct categories based on docs/10_kasane_telemetry_integration.md §5.
"""

import json
from pathlib import Path

DATASET_PATH = Path(__file__).resolve().parent.parent / "dist" / "datasets" / "kasane_nano_fused_samples.jsonl"

def make_telemetry(showing=50.0, staccato=50.0, polyphony=50.0, confession=0.0, tension=0.0, speculation=0.0, cadence=0.0, crestFactor=1.0, maxDependencyDistance=12, semanticJumps=None):
    return {
        "texture": {
            "showing": float(showing),
            "staccato": float(staccato),
            "polyphony": float(polyphony),
            "confession": float(confession),
            "tension": float(tension),
            "speculation": float(speculation),
            "cadence": float(cadence),
        },
        "dynamics": {
            "crestFactor": float(crestFactor),
            "cadence": float(cadence),
            "maxDependencyDistance": int(maxDependencyDistance),
        },
        "semanticJumps": semanticJumps or []
    }

def build_samples():
    samples = []
    idx = 1

    # =========================================================================
    # 1. fid_monologue (20 samples)
    # 15 samples: FID Monologue soft-landing (confession >= 68, tension >= 0.45)
    # 5 samples: Past recollection false-positive suppression control (confession: 70, past markers)
    # =========================================================================
    fid_active_texts = [
        ("彼女は立ち尽くした。なぜこんな仕打ちを受けねばならないのか。", 10, 20),
        ("青年は空を見上げた。どうしてあの時真実を告げられなかったのだろうか。", 10, 26),
        ("男は拳を握り締めた。まさかすべてが仕組まれた罠だったとは。", 10, 21),
        ("王女は広間を駆け抜けた。なんと無慈悲な運命が待ち受けていることか。", 11, 23),
        ("少女は涙を拭った。ここで立ち止まってたまるものか。", 9, 16),
        ("老人は目を閉じた。これ以上何を奪われねばならないのか。", 9, 19),
        ("兵士は泥に塗れた。なぜ仲間を見殺しにしなければならなかったのか。", 9, 26),
        ("影は静かに佇んでいた。果たして誰がこの結末を望んだのだろうか。", 12, 23),
        ("少年は息を潜めた。どうすればあの子を救えるというのだろう。", 9, 21),
        ("騎士は折れた剣を見つめた。誇りなど最初から存在しなかったのではないか。", 14, 25),
        ("探偵は雨の街を睨みつけた。なぜあの男は自ら死を選んだのか。", 14, 18),
        ("医師の手が震えていた。まさか手遅れだったとでもいうのか。", 11, 18),
        ("母は赤ん坊を抱き締めた。どうしてこの小さな命が狙われねばならないのか。", 12, 27),
        ("旅人は荒野に倒れ伏した。なんと広大で冷酷な世界なのだろうか。", 12, 19),
        ("作家は白紙を睨んだ。自分は何のために言葉を紡いできたのだろうか。", 10, 23),
    ]

    for text, _, _ in fid_active_texts:
        exact_lead_len = text.index("。") + 1
        exact_fid_len = len(text) - exact_lead_len
        samples.append({
            "id": f"fused-fid-{idx:03d}",
            "category": "fid_monologue",
            "subcategory": "active_fid_softlanding",
            "text": text,
            "lead_len": exact_lead_len,
            "fid_len": exact_fid_len,
            "telemetry": make_telemetry(showing=30.0, staccato=35.0, polyphony=40.0, confession=78.0, tension=0.65),
            "expected": {
                "disambiguator_lead_modality": 0, # Narrative
                "disambiguator_fid_modality": 2,  # Monologue
                "pov_violation_type": "free_indirect_discourse",
                "pov_severity": "info",
                "pov_score_penalty": 0.0
            }
        })
        idx += 1

    # 5 past recollection controls
    fid_control_texts = [
        "十年前、あの日彼女は立ち尽くしていた。なぜ逃げ出さなかったのかと悔やんでいた。",
        "あの日、幼い少年は街を離れた。どうして父の言葉を疑わなかったのかと述懐していた。",
        "当時は誰もが熱狂していた。まさか破滅が迫っていたとは思いもしなかったのだ。",
        "その頃の城下町は賑わっていた。なんと平和な時代だったのかと追憶していた。",
        "かつてあの砦には屈強な兵が集まっていた。なぜ滅び去ったのかと歴史書に記されていた。"
    ]
    for text in fid_control_texts:
        samples.append({
            "id": f"fused-fid-{idx:03d}",
            "category": "fid_monologue",
            "subcategory": "past_recollection_control",
            "text": text,
            "is_past_recollection": True,
            "telemetry": make_telemetry(showing=35.0, staccato=40.0, polyphony=45.0, confession=72.0, tension=0.55),
            "expected": {
                "disambiguator_all_modality": 0, # Should stay Narrative (no FP monologue)
                "pov_violations_count": 0
            }
        })
        idx += 1

    # =========================================================================
    # 2. epistemic_leak (20 samples)
    # 10 samples: Low showing & high speculation epistemic leakage
    # 5 samples: 5-sentence sliding window speculation spike
    # 5 samples: High showing clean mitigation controls
    # =========================================================================
    leak_direct_texts = [
        ("彼女は私を軽蔑しているに違いない。", "彼女", "軽蔑し"),
        ("上官は内心激しい怒りに震えているのだろう。", "上官", "内心"),
        ("同僚は心の内で私を嘲笑っていたに違いない。", "同僚", "心の中で"),
        ("少女は胸の内で絶望を感じたのだろう。", "少女", "胸の内で"),
        ("老人は過去の過ちを深く後悔していたに違いない。", "老人", "後悔した"),
        ("ボブは私を激しく憎んでいたに違いない。", "ボブ", "憎んだ"),
        ("警官は心の中で犯人を恐れていたのだろう。", "警官", "心の中で"),
        ("王は部下の裏切りを察知し憤っていたに違いない。", "王", "憤った"),
        ("少年は暗闇の奥に潜む怪物を恐怖したのだろう。", "少年", "恐怖した"),
        ("ライバルは私の失脚を心の中で歓喜したに違いない。", "ライバル", "心の中で"),
    ]

    for text, culprit, phrase in leak_direct_texts:
        samples.append({
            "id": f"fused-leak-{idx:03d}",
            "category": "epistemic_leak",
            "subcategory": "excessive_speculation_leak",
            "text": text,
            "culprit": culprit,
            "telemetry": make_telemetry(showing=14.0, speculation=75.0, staccato=18.0, polyphony=35.0),
            "expected": {
                "has_violation": True,
                "violation_type": "epistemic_leak_excessive_speculation",
                "severity": "warning",
                "message_contains": "【説明過多・認知漏洩の兆候】"
            }
        })
        idx += 1

    # 5 sliding window spike samples (sequences of 5 sentences where the 5th has a spike)
    spike_cases = [
        ("雨脚が次第に強くなっていった。", "木々が暗い夜風にざわめいている。", "街灯の明かりが遠くで滲んでいた。", "水たまりに足先が触れて冷たい。", "彼は私を恨んでいるに違いない。", "彼"),
        ("廊下には重苦しい沈黙が漂っていた。", "古時計の振り子だけが規則正しく時を刻む。", "埃っぽいカーテンがわずかに揺れた。", "冷たい空気が足元を這い上がってくる。", "彼女は胸中で私を嘲笑っているのだろう。", "彼女"),
        ("図書館の書架の間を歩き続けた。", "革表紙の古い本が整然と並んでいる。", "窓の外では夕暮れの鳥が鳴いていた。", "机の上には放置された羊皮紙がある。", "館長は心の中で私を疑っているに違いない。", "館長"),
        ("列車は暗闇の中を疾走していた。", "車輪の軋む単調な音が響き渡る。", "曇った窓ガラスに街の灯が流れた。", "車内には数人の乗客が眠っている。", "車掌は私の切符偽造を確信したのだろう。", "車掌"),
        ("砦の監視台から荒野を見渡した。", "砂埃が地平線の彼方に立ち上っている。", "乾いた風が石壁を打ち叩いていた。", "松明の火が頼りなく揺らめいている。", "見張り番は恐怖に怯えているに違いない。", "見張り番"),
    ]

    for s1, s2, s3, s4, s5, culprit in spike_cases:
        samples.append({
            "id": f"fused-leak-{idx:03d}",
            "category": "epistemic_leak",
            "subcategory": "sliding_window_spike",
            "sentences": [
                {"text": s1, "speculation": 15.0},
                {"text": s2, "speculation": 18.0},
                {"text": s3, "speculation": 20.0},
                {"text": s4, "speculation": 16.0},
                {"text": s5, "speculation": 78.0, "culprit": culprit},
            ],
            "expected": {
                "spike_sentence_index": 4,
                "violation_type": "epistemic_leak_excessive_speculation"
            }
        })
        idx += 1

    # 5 high showing clean mitigation controls
    clean_mitigation_texts = [
        ("彼女は唇をきつく噛み締め、悔しそうに見えた。", "彼女"),
        ("彼は顔色を蒼白にして、恐れているようだった。", "彼"),
        ("男は額に脂汗を浮かべ、焦っているような表情を見せた。", "男"),
        ("老人は肩を落とし、深く落胆した様子だった。", "老人"),
        ("少女は震える指先を握り締め、怯えているように見えた。", "少女"),
    ]
    for text, subject in clean_mitigation_texts:
        samples.append({
            "id": f"fused-leak-{idx:03d}",
            "category": "epistemic_leak",
            "subcategory": "high_showing_mitigation_clean",
            "text": text,
            "subject": subject,
            "telemetry": make_telemetry(showing=75.0, speculation=20.0, staccato=45.0),
            "expected": {
                "has_violation": False,
                "pov_consistency_score": 1.0
            }
        })
        idx += 1

    # =========================================================================
    # 3. pas_semantic_jump (20 samples)
    # 10 samples: jumpDegree < 40 normal frame boost (+gamma)
    # 10 samples: jumpDegree >= 85 extreme conceptual jump / high polyphony distance attenuation
    # =========================================================================
    pas_boost_triplets = [
        ("少女が花を摘む。", 0, 4, 1, 20),   # 少女(0) -> 摘む(4), ガ格, jump=20
        ("騎士が剣を抜いた。", 0, 4, 1, 15), # 騎士(0) -> 抜いた(4), ガ格, jump=15
        ("鳥が空を飛んでいる。", 0, 4, 1, 10),# 鳥(0) -> 飛んでいる(4), ガ格, jump=10
        ("母が料理を作る。", 0, 4, 1, 18),   # 母(0) -> 作る(4), ガ格, jump=18
        ("子供が絵を描いた。", 0, 4, 1, 12), # 子供(0) -> 描いた(4), ガ格, jump=12
        ("風が木々を揺らす。", 0, 5, 1, 25), # 風(0) -> 揺らす(5), ガ格, jump=25
        ("少年が石を投げた。", 0, 4, 1, 14), # 少年(0) -> 投げた(4), ガ格, jump=14
        ("猫が庭を駆ける。", 0, 4, 1, 22),   # 猫(0) -> 駆ける(4), ガ格, jump=22
        ("星が夜空に光る。", 0, 5, 1, 16),   # 星(0) -> 光る(5), ガ格, jump=16
        ("炎が薪を包み込む。", 0, 4, 1, 30), # 炎(0) -> 包み込む(4), ガ格, jump=30
    ]

    for text, n_idx, v_idx, c_marker, jump in pas_boost_triplets:
        samples.append({
            "id": f"fused-pas-{idx:03d}",
            "category": "pas_semantic_jump",
            "subcategory": "normal_jump_boost",
            "text": text,
            "noun_idx": n_idx,
            "verb_idx": v_idx,
            "case_marker": c_marker,
            "jump_degree": jump,
            "telemetry": make_telemetry(polyphony=45.0, semanticJumps=[{
                "utf16Start": n_idx, "utf16End": n_idx + 2,
                "nounTokenIdx": n_idx, "verbTokenIdx": v_idx,
                "caseMarker": c_marker, "jumpDegree": jump
            }]),
            "expected": {
                "score_delta_type": "boost",
                "min_score_gain": 0.75
            }
        })
        idx += 1

    pas_penalty_triplets = [
        ("概念が宇宙の深淵へと拡散した。", 0, 11, 1, 92),
        ("運命が数千年の彼方に木霊する。", 0, 11, 1, 88),
        ("悲哀が幾何学的な結晶を紡ぎ出す。", 0, 12, 1, 95),
        ("真理が時空の狭間に融解していく。", 0, 11, 1, 90),
        ("虚無が星屑の残骸を飲み込み尽くす。", 0, 12, 1, 86),
        # 5 samples with high polyphony distance attenuation (dist > 15, polyphony >= 75)
        ("遠い過去の霧深き森の奥に眠る古城の石棺から蘇った王が立ち上がる。", 0, 26, 1, 0),
        ("暗黒の彼方より来たりて世界を混沌に陥れんと企てる魔導士が詠唱する。", 0, 27, 1, 0),
        ("忘れ去られた海底都市の神殿に鎮座する巨大な偶像の眼が怪しく輝いた。", 0, 27, 1, 0),
        ("幾重にも張り巡らされた謀略の糸を巧妙に手繰り寄せる黒幕が微笑む。", 0, 26, 1, 0),
        ("冷酷なる氷河の下に閉じ込められた古代生物の封印が静かに解かれた。", 0, 26, 1, 0),
    ]

    for item in pas_penalty_triplets:
        text, n_idx, v_idx, c_marker, jump = item
        is_poly_penalty = jump == 0
        jumps = [] if is_poly_penalty else [{
            "utf16Start": n_idx, "utf16End": n_idx + 2,
            "nounTokenIdx": n_idx, "verbTokenIdx": v_idx,
            "caseMarker": c_marker, "jumpDegree": jump
        }]
        poly = 82.0 if is_poly_penalty else 40.0
        samples.append({
            "id": f"fused-pas-{idx:03d}",
            "category": "pas_semantic_jump",
            "subcategory": "poly_distance_attenuation" if is_poly_penalty else "extreme_jump_penalty",
            "text": text,
            "noun_idx": n_idx,
            "verb_idx": v_idx,
            "case_marker": c_marker,
            "jump_degree": jump,
            "telemetry": make_telemetry(polyphony=poly, semanticJumps=jumps),
            "expected": {
                "score_delta_type": "penalty",
                "max_score_delta": -0.15
            }
        })
        idx += 1

    # =========================================================================
    # 4. classical_cadence_and_irregularity (20 samples)
    # 10 samples: Cadenced classical prose with particle repetition relaxation (cadence >= 35)
    # 10 samples: Intentional stylistic irregularity soft-landing (polyphony >= 72 && cadence >= 48)
    # =========================================================================
    cadence_texts = [
        "我が輩の手のひらの上の真珠の如き露。",
        "春の夜の夢の浮橋のとだえがちに。",
        "奥山の紅葉の錦の織りなす山の端。",
        "祇園精舎の鐘の声の諸行無常の響き。",
        "東の野に炎の立つ見えて返り見すれば。",
        "秋の田のかりほの庵の苫をあらみ。",
        "久方の光のどけき春の日に静心なく。",
        "田子の浦にうち出でて見れば白妙の富士。",
        "ちはやぶる神代も聞かず竜田川からくれなゐに。",
        "いにしへの奈良の都の八重桜けふ九重に。",
    ]

    for text in cadence_texts:
        samples.append({
            "id": f"fused-lint-{idx:03d}",
            "category": "classical_cadence_and_irregularity",
            "subcategory": "cadence_particle_relaxation",
            "text": text,
            "telemetry": make_telemetry(cadence=45.0, showing=60.0, polyphony=45.0),
            "expected": {
                "particle_repetition_warning_count": 0 # Suppressed due to cadence relaxation
            }
        })
        idx += 1

    irregularity_texts = [
        ("絶望の底に沈みゆく街の片隅で、冷酷な雨に打たれながら立ち尽くす少年は、運命に抗うように拳を握り締めた。", 25, 45),
        ("漆黒の霧が立ち込める古戦場において、数多の亡霊の叫び声を背に受けた勇者は、光り輝く聖剣を高く掲げた。", 26, 46),
        ("長き歳月の果てに風化し荒れ果てた神殿の回廊で、ただ一人残された賢者は、禁断の予言書を静かに開いた。", 26, 45),
        ("幾重にも重なる謀略と裏切りが交錯する王宮の深奥で、冷徹なる宰相は、次なる布石を盤上に打った。", 24, 43),
        ("天を衝く白銀の山嶺を望む断崖絶壁の上で、吹き荒ぶ猛吹雪に耐え忍ぶ登山家は、頂への一歩を踏み出した。", 25, 46),
        ("深海数千メートルの完全なる暗黒が支配する水底で、太古の眠りから覚醒した異形の怪物は、蒼く発光した。", 26, 45),
        ("無数の歯車と蒸気が唸りを上げる巨大機関の炉の前で、煤に塗れた技師は、最後のレバーを力強く引いた。", 25, 44),
        ("夜毎怪異が蠢く呪われた森の境界面において、不気味な梟の鳴き声を耳にした旅人は、足を速めた。", 24, 42),
        ("百年の因縁に終止符を打つ決戦の場において、互いの信念を刃に宿した二人の剣士は、激しく激突した。", 25, 44),
        ("星々の煌めきさえ届かぬ虚無の宇宙空間で、漂流を続ける孤独な宇宙船は、微かな救難信号を受信した。", 25, 44),
    ]

    for text, subj_len, pred_pos in irregularity_texts:
        samples.append({
            "id": f"fused-lint-{idx:03d}",
            "category": "classical_cadence_and_irregularity",
            "subcategory": "intentional_irregularity_softlanding",
            "text": text,
            "telemetry": make_telemetry(polyphony=78.0, cadence=52.0, tension=0.6, confession=40.0, crestFactor=2.2, maxDependencyDistance=12),
            "expected": {
                "is_intentional_irregularity": True,
                "severity": "info",
                "message_contains": "【文体的破格（意図的演出）】"
            }
        })
        idx += 1

    # =========================================================================
    # 5. staccato_zero_pronoun (20 samples)
    # 15 samples: High staccato (>= 65) dialogue immediate omission speaker anchoring
    # 5 samples: Low staccato normal decay controls
    # =========================================================================
    staccato_dialogue_pairs = [
        ("「逃げろ！」", "ボブ", "走り出した。"),
        ("「撃て！」", "隊長", "引き金を引いた。"),
        ("「待ってくれ」", "アリス", "立ち止まった。"),
        ("「おい、見るな」", "ジョン", "顔を背けた。"),
        ("「早く来い」", "兄", "手を引いた。"),
        ("「火を消せ！」", "消防士", "ホースを構えた。"),
        ("「鍵を開けろ」", "警官", "扉を蹴破った。"),
        ("「諦めるな」", "コーチ", "拳を突き上げた。"),
        ("「静かにしろ」", "見張り", "物陰に潜んだ。"),
        ("「伏せろ！」", "兵士", "泥に飛び込んだ。"),
        ("「ついて来い」", "案内人", "森へ入った。"),
        ("「金を置け」", "強盗", "銃口を向けた。"),
        ("「大丈夫かい」", "医者", "脈を測った。"),
        ("「もう終わりだ」", "敵将", "剣を捨てた。"),
        ("「さあ行こう」", "少女", "笑い声をあげた。"),
    ]

    for d_text, speaker, s_text in staccato_dialogue_pairs:
        samples.append({
            "id": f"fused-zero-{idx:03d}",
            "category": "staccato_zero_pronoun",
            "subcategory": "dialogue_speaker_anchoring",
            "prev_dialogue": d_text,
            "speaker": speaker,
            "current_text": s_text,
            "telemetry": make_telemetry(staccato=75.0, showing=30.0, polyphony=25.0),
            "expected": {
                "resolved_subject": speaker,
                "distance_penalty_waived": True,
                "anchor_boost_applied": True
            }
        })
        idx += 1

    # 5 low staccato controls
    normal_decay_pairs = [
        ("「明日は晴れるだろうか」", "アリス", "窓の外をぼんやりと眺めていた。"),
        ("「お茶でも淹れましょう」", "メイド", "戸棚から陶器のカップを取り出した。"),
        ("「昔のことは忘れました」", "老人", "暖炉の火を静かに見つめ続けた。"),
        ("「旅の準備を整えねば」", "商人", "荷造りの勘定書きを何度も点検した。"),
        ("「この本は素晴らしい」", "学者", "眼鏡の位置を直して頁を繰った。"),
    ]

    for d_text, speaker, s_text in normal_decay_pairs:
        samples.append({
            "id": f"fused-zero-{idx:03d}",
            "category": "staccato_zero_pronoun",
            "subcategory": "normal_decay_control",
            "prev_dialogue": d_text,
            "speaker": speaker,
            "current_text": s_text,
            "telemetry": make_telemetry(staccato=30.0, showing=55.0, polyphony=40.0),
            "expected": {
                "is_staccato_boosted": False
            }
        })
        idx += 1

    # =========================================================================
    # 6. blind_spot_adversarial (20 samples)
    # Cold ruthless stress testing: bureaucratic text, true twist errors, double negation, POV slips
    # =========================================================================
    adversarial_samples = [
        # 1-4: True subject-predicate twists (must be flagged as warning/error)
        ("私の夢は、幼い頃から努力を重ねて世界大会で優勝したからです。", "true_twist", {"has_twist_warning": True}),
        ("彼が失敗した原因は、十分な準備を怠って油断したためでしたからです。", "true_twist", {"has_twist_warning": True}),
        ("政府の基本方針としては、経済再生を図ることを目標としたからです。", "true_twist", {"has_twist_warning": True}),
        ("彼女の提案した計画は、人員を削減して効率化を目指したからです。", "true_twist", {"has_twist_warning": True}),

        # 5-8: Double negation
        ("この計画にリスクが存在しないわけではない。", "double_negation", {"has_double_negation": True}),
        ("彼が嘘をついていると思わないでもない。", "double_negation", {"has_double_negation": True}),
        ("成功の可能性が全くなくもない。", "double_negation", {"has_double_negation": True}),
        ("その主張を全面的に否定できない。", "double_negation", {"has_double_negation": True}),

        # 9-12: Bureaucratic extremely long legal prose (adversarial stress)
        ("本規程は、機構における情報資産の適切な管理及び運用を確保し、もって業務の円滑な遂行に資することを目的とする。", "bureaucratic_legal", {"expected_pass": True}),
        ("受託者は、委託業務の遂行に際して知り得た秘密を、第三者に漏洩し、又は他の目的に使用してはならない。", "bureaucratic_legal", {"expected_pass": True}),
        ("当事者は、前項の規定にかかわらず、書面による合意がある場合に限り、本契約の全部又は一部を解除することができる。", "bureaucratic_legal", {"expected_pass": True}),
        ("甲は、乙の責めに帰すべき事由により損害を被ったときは、乙に対して損害賠償を請求することができる。", "bureaucratic_legal", {"expected_pass": True}),

        # 13-16: Jukujikun / Literary idiomatic edge cases
        ("五月雨が降り続く軒下で、時雨の音に耳を傾けていた。", "jukujikun", {"expected_pass": True}),
        ("大和魂を胸に秘めた男は、紅葉狩りの群衆を避けて歩いた。", "jukujikun", {"expected_pass": True}),
        ("昨日今日と吹き荒れた野分が、庭の百日紅の花を散らした。", "jukujikun", {"expected_pass": True}),
        ("鍛冶屋の煙突から立ち上る煙が、雲雀の囀る青空に溶けていく。", "jukujikun", {"expected_pass": True}),

        # 17-20: Subtle POV slip / boundary leakage (camera eye mode leaks)
        ("部屋の中には二人の男が座っていた。警官は後悔した。", "objective_camera_leak", {"has_objective_leak": True}),
        ("法廷は静寂に包まれていた。証人は内心で怯えていた。", "objective_camera_leak", {"has_objective_leak": True}),
        ("駅のベンチに座る旅行者。誰もが孤独を実感していた。", "objective_camera_leak", {"has_objective_leak": True}),
        ("街角を行き交う群衆。男は胸の内で悪巧みを巡らせていた。", "objective_camera_leak", {"has_objective_leak": True}),
    ]

    for text, subcat, exp in adversarial_samples:
        samples.append({
            "id": f"fused-adv-{idx:03d}",
            "category": "blind_spot_adversarial",
            "subcategory": subcat,
            "text": text,
            "telemetry": make_telemetry(showing=40.0, staccato=40.0, polyphony=50.0),
            "expected": exp
        })
        idx += 1

    return samples

def main():
    samples = build_samples()
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Successfully generated {len(samples)} verified samples to {DATASET_PATH}")
    
    # Category summary
    categories = {}
    for s in samples:
        cat = s["category"]
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat, count in categories.items():
        print(f"  - {cat}: {count} samples")

if __name__ == "__main__":
    main()
