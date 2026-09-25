/**
 * @file CharTokenizer.ts
 * Fixed 2,048 vocabulary character-level tokenizer specialized for literary Japanese text.
 * Ensures zero undefined words/tokens and 1-to-1 bidirectional mapping between
 * editor text offsets (UTF-16 / Unicode CodePoint) and tensor token indices.
 */

export const VOCAB_SIZE = 2048;

export const SPECIAL_TOKENS = {
  PAD: '[PAD]',
  UNK: '[UNK]',
  BOS: '[BOS]',
  EOS: '[EOS]',
  MASK: '[MASK]',
  CLS: '[CLS]',
  SEP: '[SEP]',
  RESERVED: '[RESERVED]',
} as const;

export interface TokenOffset {
  /** Index in tensor tokens array */
  tensorIndex: number;
  /** UTF-16 code unit start offset in editor text (inclusive) */
  utf16Start: number;
  /** UTF-16 code unit end offset in editor text (exclusive) */
  utf16End: number;
  /** Unicode CodePoint start offset in editor text (inclusive) */
  codePointStart: number;
  /** Unicode CodePoint end offset in editor text (exclusive) */
  codePointEnd: number;
  /** Original character in input text */
  originalChar: string;
  /** Normalized or projected character used for tokenization */
  tokenChar: string;
  /** Token ID in vocabulary (0..2047) */
  tokenId: number;
  /** Whether character was mapped as OOV fallback */
  isOov: boolean;
}

export interface EncodeResult {
  tokens: number[];
  offsets: TokenOffset[];
}

export interface EncodeOptions {
  addBos?: boolean;
  addEos?: boolean;
}

/** Map of variant kanji and halfwidth katakana to canonical vocabulary form */
const VARIANT_MAP: Record<string, string> = {
  '髙': '高',
  '﨑': '崎',
  '塚': '塚',
  '鷗': '鴎',
  '凜': '凛',
  '龍': '竜',
  '櫻': '桜',
  '萬': '万',
  '國': '国',
  '邊': '辺',
  '邉': '辺',
  '廣': '広',
  '澤': '沢',
  '齋': '斎',
  '齊': '斉',
  '舊': '旧',
  '醫': '医',
  '瀧': '滝',
  '藪': '薮',
  '鷄': '鶏',
  '神': '神',
  '福': '福',
  '禮': '礼',
  '社': '社',
  '羽': '羽',
  '淸': '清',
  '德': '徳',
  '都': '都',
  '晴': '晴',
  '祥': '祥',
  'ｱ': 'ア', 'ｲ': 'イ', 'ｳ': 'ウ', 'ｴ': 'エ', 'ｵ': 'オ',
  'ｶ': 'カ', 'ｷ': 'キ', 'ｸ': 'ク', 'ｹ': 'ケ', 'ｺ': 'コ',
  'ｻ': 'サ', 'ｼ': 'シ', 'ｽ': 'ス', 'ｾ': 'セ', 'ｿ': 'ソ',
  'ﾀ': 'タ', 'ﾁ': 'チ', 'ﾂ': 'ツ', 'ﾃ': 'テ', 'ﾄ': 'ト',
  'ﾅ': 'ナ', 'ﾆ': 'ニ', 'ﾇ': 'ヌ', 'ﾈ': 'ネ', 'ﾉ': 'ノ',
  'ﾊ': 'ハ', 'ﾋ': 'ヒ', 'ﾌ': 'フ', 'ﾍ': 'ヘ', 'ﾎ': 'ホ',
  'ﾏ': 'マ', 'ﾐ': 'ミ', 'ﾑ': 'ム', 'ﾒ': 'メ', 'ﾓ': 'モ',
  'ﾔ': 'ヤ', 'ﾕ': 'ユ', 'ﾖ': 'ヨ',
  'ﾗ': 'ラ', 'ﾘ': 'リ', 'ﾙ': 'ル', 'ﾚ': 'レ', 'ﾛ': 'ロ',
  'ﾜ': 'ワ', 'ｦ': 'ヲ', 'ﾝ': 'ン', 'ｯ': 'ッ', 'ｬ': 'ヤ', 'ｭ': 'ユ', 'ｮ': 'ヨ',
  'ｰ': 'ー',
};

/** 2,048 fixed vocabulary array */
const VOCAB_TABLE: string[] = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]", "[CLS]", "[SEP]", "[RESERVED]", "\n", "\t", " ", "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?", "@", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "[", "\\", "]", "^", "_", "`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "　", "、", "。", "・", "…", "―", "ー", "！", "？", "「", "」", "『", "』", "（", "）", "【", "】", "〈", "〉", "《", "》", "〔", "〕", "［", "］", "｛", "｝", "＜", "＞", "：", "；", "＋", "－", "＝", "＊", "％", "＃", "＠", "＆", "｜", "／", "￥", "‘", "’", "“", "”", "〜", "‥", "ゝ", "ゞ", "ヽ", "ヾ", "々", "〇", "〆", "※", "〒", "→", "←", "↑", "↓", "◆", "◇", "■", "□", "▲", "△", "▼", "▽", "★", "☆", "♪", "†", "‡", "ぁ", "あ", "ぃ", "い", "ぅ", "う", "ぇ", "え", "ぉ", "お", "か", "が", "き", "ぎ", "く", "ぐ", "け", "げ", "こ", "ご", "さ", "ざ", "し", "じ", "す", "ず", "せ", "ぜ", "そ", "ぞ", "た", "だ", "ち", "ぢ", "っ", "つ", "づ", "て", "で", "と", "ど", "な", "に", "ぬ", "ね", "の", "は", "ば", "ぱ", "ひ", "び", "ぴ", "ふ", "ぶ", "ぷ", "へ", "べ", "ぺ", "ほ", "ぼ", "ぽ", "ま", "み", "む", "め", "も", "ゃ", "や", "ゅ", "ゆ", "ょ", "よ", "ら", "り", "る", "れ", "ろ", "ゎ", "わ", "ゐ", "ゑ", "を", "ん", "ゔ", "ゕ", "ゖ", "ァ", "ア", "ィ", "イ", "ゥ", "ウ", "ェ", "エ", "ォ", "オ", "カ", "ガ", "キ", "ギ", "ク", "グ", "ケ", "ゲ", "コ", "ゴ", "サ", "ザ", "シ", "ジ", "ス", "ズ", "セ", "ゼ", "ソ", "ゾ", "タ", "ダ", "チ", "ヂ", "ッ", "ツ", "ヅ", "テ", "デ", "ト", "ド", "ナ", "ニ", "ヌ", "ネ", "ノ", "ハ", "バ", "パ", "ヒ", "ビ", "ピ", "フ", "ブ", "プ", "ヘ", "ベ", "ペ", "ホ", "ボ", "ポ", "マ", "ミ", "ム", "メ", "モ", "ャ", "ヤ", "ュ", "ユ", "ョ", "ヨ", "ラ", "リ", "ル", "レ", "ロ", "ヮ", "ワ", "ヰ", "ヱ", "ヲ", "ン", "ヴ", "ヵ", "ヶ", "ヷ", "ヸ", "ヹ", "ヺ", "吾", "輩", "猫", "私", "僕", "俺", "君", "彼", "女", "名", "前", "無", "高", "崎", "塚", "鴎", "凛", "蓮", "葵", "翔", "咲", "桜", "茜", "楓", "雫", "颯", "紬", "凪", "瑛", "澪", "杏", "結", "芽", "萌", "遥", "蒼", "隼", "拓", "悠", "駿", "龍", "耀", "琉", "涼", "栞", "奈", "莉", "乃", "華", "冴", "渚", "絢", "綾", "霞", "桂", "瞳", "朔", "爽", "菫", "翠", "鳳", "雅", "暁", "瑠", "璃", "緋", "紫", "琥", "珀", "殺", "闇", "光", "影", "剣", "魔", "法", "勇", "王", "神", "姫", "皇", "帝", "騎", "士", "妖", "精", "一", "右", "雨", "円", "音", "下", "火", "花", "貝", "学", "気", "九", "休", "玉", "金", "空", "月", "犬", "見", "口", "校", "左", "三", "山", "子", "四", "糸", "字", "耳", "七", "車", "手", "十", "出", "小", "上", "森", "人", "水", "正", "生", "青", "夕", "石", "赤", "千", "川", "先", "早", "足", "村", "大", "男", "竹", "中", "虫", "町", "天", "田", "土", "二", "日", "入", "年", "白", "八", "百", "文", "木", "本", "目", "立", "力", "林", "六", "引", "羽", "雲", "園", "遠", "黄", "何", "科", "夏", "家", "歌", "画", "回", "会", "海", "絵", "外", "角", "楽", "活", "間", "丸", "岩", "顔", "汽", "記", "帰", "弓", "牛", "魚", "京", "強", "教", "近", "兄", "形", "計", "元", "言", "原", "戸", "古", "午", "後", "語", "工", "公", "広", "交", "考", "行", "合", "谷", "国", "黒", "今", "才", "細", "作", "算", "止", "市", "矢", "姉", "思", "紙", "寺", "自", "時", "室", "社", "弱", "首", "秋", "週", "春", "書", "少", "場", "色", "食", "心", "新", "親", "図", "数", "西", "声", "星", "晴", "切", "雪", "船", "線", "組", "走", "多", "太", "体", "台", "地", "池", "知", "茶", "昼", "長", "鳥", "朝", "直", "通", "弟", "店", "点", "電", "刀", "冬", "当", "東", "答", "頭", "同", "道", "読", "内", "南", "肉", "馬", "買", "売", "麦", "半", "番", "父", "風", "分", "章", "用", "曜", "来", "理", "話", "悪", "安", "暗", "医", "委", "意", "育", "員", "院", "飲", "運", "泳", "駅", "央", "横", "屋", "温", "化", "荷", "界", "開", "階", "寒", "感", "漢", "館", "岸", "起", "期", "客", "究", "急", "級", "宮", "球", "去", "橋", "業", "曲", "局", "銀", "区", "苦", "具", "係", "軽", "血", "決", "研", "県", "庫", "湖", "向", "幸", "港", "号", "根", "祭", "皿", "仕", "死", "使", "始", "指", "歯", "詩", "次", "事", "持", "式", "実", "写", "者", "主", "取", "守", "酒", "受", "州", "拾", "終", "集", "重", "宿", "所", "暑", "助", "昭", "消", "商", "勝", "乗", "植", "申", "身", "真", "深", "進", "世", "整", "昔", "全", "相", "送", "想", "息", "族", "他", "打", "対", "待", "代", "第", "題", "炭", "短", "談", "着", "注", "柱", "丁", "帳", "調", "追", "定", "庭", "笛", "鉄", "転", "都", "度", "投", "豆", "島", "湯", "登", "等", "動", "童", "農", "波", "配", "倍", "箱", "畑", "発", "反", "坂", "板", "皮", "妃", "匹", "樋", "表", "秒", "病", "品", "負", "部", "服", "福", "物", "平", "返", "勉", "弁", "保", "枚", "役", "薬", "由", "油", "有", "遊", "予", "羊", "洋", "葉", "陽", "様", "落", "流", "旅", "両", "緑", "礼", "和", "愛", "案", "以", "衣", "位", "囲", "胃", "印", "英", "栄", "塩", "億", "加", "果", "貨", "課", "改", "械", "害", "街", "各", "覚", "完", "官", "管", "関", "観", "願", "希", "季", "紀", "喜", "旗", "器", "機", "議", "求", "泣", "救", "給", "清", "挙", "漁", "共", "協", "鏡", "競", "極", "訓", "軍", "郡", "径", "型", "景", "芸", "欠", "建", "健", "験", "固", "功", "好", "康", "航", "告", "差", "菜", "最", "材", "昨", "札", "刷", "察", "参", "産", "散", "残", "氏", "試", "児", "治", "滋", "辞", "鹿", "失", "借", "種", "周", "祝", "順", "初", "松", "笑", "唱", "焼", "照", "城", "縄", "成", "静", "説", "折", "浅", "単", "戦", "選", "然", "争", "倉", "巣", "束", "側", "続", "卒", "孫", "帯", "隊", "達", "置", "典", "伝", "灯", "働", "特", "徳", "毒", "熱", "念", "信", "不", "付", "夫", "府", "副", "粉", "兵", "別", "辺", "変", "便", "包", "牧", "博", "飯", "飛", "必", "票", "標", "圧", "移", "因", "永", "営", "衛", "易", "益", "液", "演", "応", "往", "可", "仮", "価", "過", "賀", "快", "解", "格", "確", "額", "刊", "幹", "慣", "眼", "基", "寄", "規", "技", "義", "逆", "久", "旧", "居", "境", "均", "禁", "句", "群", "経", "潔", "件", "券", "険", "検", "限", "現", "減", "故", "個", "護", "効", "厚", "耕", "鉱", "構", "興", "講", "再", "妻", "採", "際", "在", "財", "罪", "雑", "酸", "賛", "支", "志", "枝", "師", "資", "飼", "示", "似", "識", "舎", "謝", "煮", "射", "捨", "釈", "授", "修", "述", "術", "準", "序", "除", "承", "招", "証", "条", "状", "情", "常", "織", "職", "制", "性", "政", "勢", "製", "税", "責", "績", "接", "設", "舌", "絶", "銭", "祖", "素", "総", "造", "像", "増", "則", "測", "属", "率", "損", "退", "貸", "態", "団", "断", "築", "張", "提", "程", "適", "敵", "統", "銅", "導", "独", "任", "能", "破", "犯", "判", "版", "比", "肥", "非", "備", "費", "批", "秘", "婦", "富", "布", "武", "復", "複", "仏", "編", "墓", "報", "豊", "防", "貿", "暴", "脈", "務", "夢", "迷", "綿", "模", "訳", "預", "容", "留", "領", "異", "遺", "域", "宇", "映", "延", "沿", "我", "灰", "拡", "革", "閣", "割", "株", "干", "巻", "看", "簡", "危", "机", "揮", "貴", "疑", "吸", "供", "胸", "郷", "勤", "筋", "系", "敬", "警", "劇", "穴", "絹", "権", "憲", "源", "厳", "己", "呼", "誤", "后", "孝", "紅", "降", "鋼", "刻", "穀", "骨", "困", "砂", "座", "済", "裁", "策", "冊", "姿", "至", "視", "詞", "誌", "磁", "尺", "若", "樹", "収", "宗", "就", "衆", "従", "縦", "縮", "熟", "純", "処", "署", "諸", "将", "傷", "障", "蒸", "針", "仁", "垂", "推", "寸", "盛", "聖", "誠", "宣", "専", "泉", "洗", "染", "善", "創", "奏", "層", "操", "蔵", "臓", "存", "尊", "展", "探", "誕", "段", "暖", "値", "宙", "忠", "著", "庁", "頂", "腸", "潮", "賃", "痛", "討", "党", "糖", "得", "届", "難", "乳", "認", "納", "脳", "派", "拝", "背", "肺", "俳", "班", "晩", "否", "腹", "奮", "並", "陛", "閉", "片", "補", "暮", "宝", "訪", "亡", "忘", "棒", "幕", "密", "盟", "郵", "優", "幼", "欲", "翌", "乱", "卵", "覧", "裏", "律", "臨", "朗", "論", "丂", "丄", "丅", "丆", "万", "丈", "丌", "与", "丏", "丐", "丑", "丒", "专", "且", "丕", "丗", "丘", "丙", "业", "丛", "东", "丝", "丞", "丟", "丠", "丢", "丣", "两", "严", "丧", "丨", "丩", "个", "丫", "丬", "丮", "丯", "丰", "丱", "串", "丳", "临", "丵", "丶", "丷", "丹", "为", "丼", "丽", "举", "丿", "乀", "乁", "乂", "乄", "乆", "乇", "么", "义", "乊", "之", "乌", "乍", "乎", "乏", "乐", "乑", "乒", "乓", "乔", "乕", "乖", "乘", "乙", "乚", "乛", "乜", "乞", "也", "习", "乡", "乢", "乣", "乤", "乥", "书", "乧", "乨", "乩", "乪", "乫", "乬", "乭", "乮", "乯", "买", "乲", "乴", "乵", "乶", "乷", "乸", "乹", "乺", "乻", "乼", "乽", "乾", "乿", "亀", "亁", "亂", "亃", "亄", "亅", "了", "亇", "亊", "亍", "于", "亏", "亐", "云", "互", "亓", "五", "井", "亖", "亗", "亘", "亙", "亚", "些", "亜", "亝", "亞", "亟", "亠", "亢", "亣", "亥", "亦", "产", "亨", "亩", "亪", "享", "亭", "亮", "亯", "亰", "亱", "亲", "亳", "亴", "亵", "亶", "亷", "亸", "亹", "亻", "亼", "亽", "亾", "亿", "什", "仂", "仃", "仄", "仅", "仆", "仇", "仈", "仉", "介", "仌", "仍", "从", "仐", "仑", "仒", "仓", "仔", "仗", "仙", "仚", "仛", "仜", "仝", "仞", "仟", "仠", "仡", "仢", "令", "仦", "仧", "仨", "仩", "仪", "仫", "们", "仭", "仯", "仰", "仱", "仲", "仳", "仴", "仵", "价", "仸", "仹", "仺", "仼", "份", "仾", "仿", "伀", "企", "伂", "伃", "伄", "伅", "伆", "伇", "伈", "伉", "伊", "伋", "伌", "伍", "伎", "伏", "伐", "伒", "伓", "伔", "伕", "伖", "众", "优", "伙", "伛", "伜", "伞", "伟", "传", "伡", "伢", "伣", "伤", "伥", "伦", "伧", "伨", "伩", "伪", "伫", "伬", "伭", "伮", "伯", "估", "伱", "伲", "伳", "伴", "伵", "伶", "伷", "伸", "伹", "伺", "伻", "伽", "伾", "伿", "佀", "佁", "佂", "佃", "佄", "佅", "但", "佇", "佈", "佉", "佊", "佋", "佌", "低", "住", "佐", "佑", "佒", "佔", "佖", "佗", "佘", "余", "佚", "佛", "佝", "佞", "佟", "你", "佡", "佢", "佣", "佤", "佥", "佦", "佧", "佨", "佩", "佪", "佫", "佬", "佭", "佮", "佯", "佰", "佱", "佲", "佳", "佴", "併", "佶", "佷", "佸", "佹", "佺", "佻", "佼", "佽", "佾", "侀", "侁", "侂", "侃", "侄", "侅", "來", "侇", "侈", "侉", "侊", "例", "侌", "侍", "侎", "侏", "侐", "侑", "侒", "侓", "侔", "侕", "侖", "侗", "侘", "侙", "侚", "侜", "依", "侞", "侟", "侠", "侢", "侣", "侤", "侥", "侦", "侧", "侨", "侩", "侪", "侫", "侬", "侭", "侮", "侯", "侰", "侱", "侲", "侳", "侴", "侵", "侶", "侷", "侸", "侹", "侺", "侻", "侼", "侽", "侾", "俀", "俁", "促", "俄", "俅", "俆", "俇", "俈", "俉", "俊", "俋", "俌", "俍", "俎", "俏", "俐", "俑", "俒", "俓", "俔", "俕", "俖", "俗", "俘", "俙", "俚", "俛", "俜", "俞", "俟", "俠", "俢", "俣", "俤", "俥", "俦", "俧", "俨", "俩", "俪", "俫", "俬", "俭", "俯", "俰", "俱", "俲", "俴", "俵", "俶", "俷", "俸", "俹", "俻", "俼", "俽", "俾", "俿", "倀", "倁", "倂", "倃", "倄", "倅", "倆", "倇", "倈", "倊", "倌", "倎", "倏", "倐", "們", "倒", "倓", "倔", "倕", "倖", "倗", "倘", "候", "倚", "倛", "倜", "倝", "倞", "倠", "倡", "倢", "倣", "倥", "倦", "倧", "倨", "倩", "倪", "倫", "倬", "倭", "倮", "倯", "倰", "倱", "倲", "倳", "倴", "倵", "倶", "倷", "倸", "倹", "债", "倻", "值", "倽", "倾", "倿", "偀", "偁", "偂", "偃", "偄", "偅", "偆", "假", "偈", "偉", "偊", "偋", "偌", "偍", "偎", "偏", "偐", "偑", "偒", "偓", "偔", "偕", "偖", "偗", "偘", "偙", "做", "偛", "停", "偝", "偞", "偟", "偠", "偡", "偢", "偣", "偤", "偦", "偧", "偨", "偩", "偪", "偫", "偬", "偭", "偮", "偯", "偰", "偱", "偲", "偳", "偵", "偶", "偷", "偸", "偹", "偺", "偻", "偼", "偽", "偾", "偿", "傀", "傁", "傂", "傃", "傄", "傅", "傆", "傇", "傈", "傉", "傊", "傋", "傌", "傍", "傎", "傏", "傐", "傑", "傒", "傓", "傔", "傕", "傖", "傗", "傘", "傚", "傛", "傜", "傝", "傞", "傟", "傠", "傡", "傢", "傣", "傤", "傥", "傦", "傧", "储", "傩", "傪", "傫", "催", "傭", "傮", "傯", "傰", "傱", "傲", "傳", "傴", "債", "傶", "傸", "傹", "傺", "傻", "傼", "傽", "傾", "傿", "僀", "僁", "僂", "僃", "僄", "僅", "僆", "僇", "僈", "僉", "僊", "僋", "僌", "僎", "僐", "僑", "僒", "僓", "僔", "僖", "僗", "僘", "僙", "僚", "僛", "僜", "僝", "僞", "僟", "僠", "僡", "僢", "僣", "僤", "僥", "僦", "僧", "僨", "僩", "僪", "僫", "僬", "僭", "僮", "僯", "僰", "僱", "僲", "僳", "僴", "僵", "僶", "僷", "僸", "價", "僺", "僻", "僼", "僽", "僾", "僿", "儀", "儁", "儂", "儃", "儅", "儆", "儇", "儈", "儉", "儊", "儋", "儌", "儍"];

/** Inverted map: char -> tokenId */
const VOCAB_MAP = new Map<string, number>();
VOCAB_TABLE.forEach((char, idx) => {
  VOCAB_MAP.set(char, idx);
});

export class CharTokenizer {
  public static readonly VOCAB_SIZE = VOCAB_SIZE;
  public static readonly PAD_TOKEN_ID = 0;
  public static readonly UNK_TOKEN_ID = 1;
  public static readonly BOS_TOKEN_ID = 2;
  public static readonly EOS_TOKEN_ID = 3;
  public static readonly MASK_TOKEN_ID = 4;
  public static readonly CLS_TOKEN_ID = 5;
  public static readonly SEP_TOKEN_ID = 6;

  /**
   * Returns the total vocabulary size (always 2048).
   */
  public getVocabSize(): number {
    return VOCAB_SIZE;
  }

  /**
   * Returns a copy of the vocabulary map (char -> tokenId).
   */
  public getVocab(): Map<string, number> {
    return new Map(VOCAB_MAP);
  }

  /**
   * Returns a copy of the vocabulary array.
   */
  public getVocabTable(): string[] {
    return [...VOCAB_TABLE];
  }

  /**
   * Normalizes a character, applying NFKC normalization and variant kanji / glyph approximation.
   */
  public normalizeChar(char: string): string {
    if (!char) return '';
    // Check direct match in vocabulary first
    if (VOCAB_MAP.has(char)) {
      return char;
    }
    // Check known variant mapping
    if (VARIANT_MAP[char] && VOCAB_MAP.has(VARIANT_MAP[char])) {
      return VARIANT_MAP[char];
    }
    // Apply Unicode NFKC normalization
    const nfkc = char.normalize('NFKC');
    if (VOCAB_MAP.has(nfkc)) {
      return nfkc;
    }
    if (VARIANT_MAP[nfkc] && VOCAB_MAP.has(VARIANT_MAP[nfkc])) {
      return VARIANT_MAP[nfkc];
    }
    // Fallback if normalized version has multiple characters (pick first if in vocab)
    for (const subChar of nfkc) {
      if (VOCAB_MAP.has(subChar)) {
        return subChar;
      }
    }

    // Safe fallback token character
    return SPECIAL_TOKENS.UNK;
  }

  /**
   * Gets the token ID for a single character (0..2047).
   */
  public getTokenId(char: string): number {
    if (VOCAB_MAP.has(char)) {
      return VOCAB_MAP.get(char)!;
    }
    const normalized = this.normalizeChar(char);
    return VOCAB_MAP.get(normalized) ?? CharTokenizer.UNK_TOKEN_ID;
  }

  /**
   * Gets the character representation for a given token ID.
   */
  public getChar(tokenId: number): string {
    if (tokenId >= 0 && tokenId < VOCAB_TABLE.length) {
      return VOCAB_TABLE[tokenId];
    }
    return SPECIAL_TOKENS.UNK;
  }

  /**
   * Encodes text into token IDs and offset mappings.
   * Guarantees zero undefined token IDs (all IDs in range [0, 2047]).
   */
  public encode(text: string, options: EncodeOptions = {}): EncodeResult {
    const tokens: number[] = [];
    const offsets: TokenOffset[] = [];

    let utf16Offset = 0;
    let codePointOffset = 0;

    if (options.addBos) {
      tokens.push(CharTokenizer.BOS_TOKEN_ID);
      offsets.push({
        tensorIndex: 0,
        utf16Start: 0,
        utf16End: 0,
        codePointStart: 0,
        codePointEnd: 0,
        originalChar: '',
        tokenChar: SPECIAL_TOKENS.BOS,
        tokenId: CharTokenizer.BOS_TOKEN_ID,
        isOov: false,
      });
    }

    // Iterate over Unicode code points using `for...of`
    for (const char of text) {
      const utf16Len = char.length; // 1 for BMP, 2 for surrogate pairs
      const codePointLen = 1;

      const normalized = this.normalizeChar(char);
      const inVocabDirectly = VOCAB_MAP.has(char);
      const tokenId = VOCAB_MAP.get(normalized) ?? CharTokenizer.UNK_TOKEN_ID;

      const tensorIndex = tokens.length;
      tokens.push(tokenId);

      offsets.push({
        tensorIndex,
        utf16Start: utf16Offset,
        utf16End: utf16Offset + utf16Len,
        codePointStart: codePointOffset,
        codePointEnd: codePointOffset + codePointLen,
        originalChar: char,
        tokenChar: normalized,
        tokenId,
        isOov: !inVocabDirectly,
      });

      utf16Offset += utf16Len;
      codePointOffset += codePointLen;
    }

    if (options.addEos) {
      const tensorIndex = tokens.length;
      tokens.push(CharTokenizer.EOS_TOKEN_ID);
      offsets.push({
        tensorIndex,
        utf16Start: utf16Offset,
        utf16End: utf16Offset,
        codePointStart: codePointOffset,
        codePointEnd: codePointOffset,
        originalChar: '',
        tokenChar: SPECIAL_TOKENS.EOS,
        tokenId: CharTokenizer.EOS_TOKEN_ID,
        isOov: false,
      });
    }

    return { tokens, offsets };
  }

  /**
   * Decodes an array of token IDs back into a string.
   * Special control tokens ([PAD], [CLS], [SEP], [BOS], [EOS], [MASK]) are omitted by default unless specified.
   */
  public decode(tokens: ArrayLike<number>, includeSpecialTokens = false): string {
    const result: string[] = [];
    for (let i = 0; i < tokens.length; i++) {
      const id = tokens[i];
      if (id < 0 || id >= VOCAB_TABLE.length) {
        continue;
      }
      const char = VOCAB_TABLE[id];
      if (!includeSpecialTokens && Object.values(SPECIAL_TOKENS).includes(char as any)) {
        continue;
      }
      result.push(char);
    }
    return result.join('');
  }

  /**
   * Maps a UTF-16 code unit offset in editor text to a tensor token index.
   */
  public utf16OffsetToTensorIndex(utf16Offset: number, offsets: TokenOffset[]): number {
    if (offsets.length === 0) return 0;
    if (utf16Offset <= offsets[0].utf16Start) return offsets[0].tensorIndex;

    const last = offsets[offsets.length - 1];
    if (utf16Offset >= last.utf16End) return last.tensorIndex;

    for (const offset of offsets) {
      if (utf16Offset >= offset.utf16Start && utf16Offset < offset.utf16End) {
        return offset.tensorIndex;
      }
    }
    return last.tensorIndex;
  }

  /**
   * Maps a tensor token index to UTF-16 code unit start and end offsets in editor text.
   */
  public tensorIndexToUtf16Offset(tensorIndex: number, offsets: TokenOffset[]): { start: number; end: number } {
    if (offsets.length === 0) return { start: 0, end: 0 };
    const clampedIndex = Math.max(0, Math.min(tensorIndex, offsets.length - 1));
    const offset = offsets[clampedIndex];
    return { start: offset.utf16Start, end: offset.utf16End };
  }

  /**
   * Maps a Unicode CodePoint offset in editor text to a tensor token index.
   */
  public codePointOffsetToTensorIndex(codePointOffset: number, offsets: TokenOffset[]): number {
    if (offsets.length === 0) return 0;
    if (codePointOffset <= offsets[0].codePointStart) return offsets[0].tensorIndex;

    const last = offsets[offsets.length - 1];
    if (codePointOffset >= last.codePointEnd) return last.tensorIndex;

    for (const offset of offsets) {
      if (codePointOffset >= offset.codePointStart && codePointOffset < offset.codePointEnd) {
        return offset.tensorIndex;
      }
    }
    return last.tensorIndex;
  }

  /**
   * Maps a tensor token index to Unicode CodePoint start and end offsets in editor text.
   */
  public tensorIndexToCodePointOffset(tensorIndex: number, offsets: TokenOffset[]): { start: number; end: number } {
    if (offsets.length === 0) return { start: 0, end: 0 };
    const clampedIndex = Math.max(0, Math.min(tensorIndex, offsets.length - 1));
    const offset = offsets[clampedIndex];
    return { start: offset.codePointStart, end: offset.codePointEnd };
  }
}
