import type { StateDeltaEvent, StateDeltaAction } from '@worldcraft/schema';
import type { PASPredicateRelation, PASArgument, PASCase } from './BiaffinePASHead.js';
import type { ZeroPronounResolutionResult } from './ZeroPronounResolver.js';

export type StandardAction = 'Acquire' | 'Drop' | 'Move' | 'ChangeState';

/**
 * Normalization dictionary table mapping Japanese predicates/stems to standard actions.
 */
export const PREDICATE_ACTION_DICTIONARY: Record<string, StandardAction> = {
  // Acquire (手に入る, 拾う, 獲得する etc.)
  '拾う': 'Acquire',
  '拾いあげる': 'Acquire',
  '拾い上げる': 'Acquire',
  '手に入れる': 'Acquire',
  '獲得する': 'Acquire',
  '獲得': 'Acquire',
  '買い取る': 'Acquire',
  '奪う': 'Acquire',
  '得る': 'Acquire',
  '受け取る': 'Acquire',
  'つかむ': 'Acquire',
  '掴む': 'Acquire',
  '捕まえる': 'Acquire',
  '入手': 'Acquire',

  // Drop (手放す, 捨てる, 渡す, 失う etc.)
  '手放す': 'Drop',
  '捨てる': 'Drop',
  '落とす': 'Drop',
  '渡す': 'Drop',
  '手渡す': 'Drop',
  '譲る': 'Drop',
  '失う': 'Drop',
  '置く': 'Drop',
  '放す': 'Drop',
  '奪われる': 'Drop',
  '喪失': 'Drop',

  // Move (向かう, 移動する, 行く etc.)
  '向かう': 'Move',
  '移動する': 'Move',
  '移動': 'Move',
  '行く': 'Move',
  '走る': 'Move',
  '逃げる': 'Move',
  '赴く': 'Move',
  '歩く': 'Move',
  '訪れる': 'Move',
  '着く': 'Move',
  '辿り着く': 'Move',
  '立ち去る': 'Move',
  '登る': 'Move',
  '降りる': 'Move',
  '去る': 'Move',
  '進む': 'Move',

  // ChangeState (倒す, 破滅させる, 壊す, 変身する etc.)
  '倒す': 'ChangeState',
  '倒れる': 'ChangeState',
  '壊す': 'ChangeState',
  '壊れる': 'ChangeState',
  '傷つける': 'ChangeState',
  '変身する': 'ChangeState',
  '死ぬ': 'ChangeState',
  '眠る': 'ChangeState',
  '覚める': 'ChangeState',
  '変化する': 'ChangeState',
  '破滅させる': 'ChangeState',
  '殺す': 'ChangeState',
  '撃つ': 'ChangeState',
  '斬る': 'ChangeState',
};

/**
 * Stem and keyword fallback rules for verb normalization
 */
const STEM_FALLBACK_RULES: Array<{ keywords: string[]; action: StandardAction }> = [
  { keywords: ['拾', '手に入', '獲得', '買', '奪', '得', '受取', '入手'], action: 'Acquire' },
  { keywords: ['捨', '落と', '手渡', '渡', '譲', '失', '置', '放', '喪失'], action: 'Drop' },
  { keywords: ['向か', '移動', '行', '走', '逃', '赴', '歩', '訪', '着', '辿り着', '登', '降', '進'], action: 'Move' },
  { keywords: ['倒', '壊', '傷', '変身', '死', '眠', '覚', '変化', '破滅', '殺', '撃', '斬'], action: 'ChangeState' },
];

export interface EventMapperOptions {
  eventIdGenerator?: () => string;
  timestamp?: number;
  customDictionary?: Record<string, StateDeltaAction>;
  defaultEntityId?: string;
  defaultAction?: StateDeltaAction;
}

/**
 * Normalizes a natural language predicate text to a standard StateDeltaAction code.
 */
export function normalizePredicateAction(
  predicateText: string,
  customDictionary?: Record<string, StateDeltaAction>,
  defaultAction: StateDeltaAction = 'ChangeState'
): StateDeltaAction {
  if (!predicateText) return defaultAction;

  const trimmed = predicateText.trim();

  // 1. Check custom dictionary
  if (customDictionary && customDictionary[trimmed]) {
    return customDictionary[trimmed];
  }

  // 2. Exact match in standard dictionary
  if (PREDICATE_ACTION_DICTIONARY[trimmed]) {
    return PREDICATE_ACTION_DICTIONARY[trimmed];
  }

  // 3. Keyword / stem matching
  for (const rule of STEM_FALLBACK_RULES) {
    for (const kw of rule.keywords) {
      if (trimmed.includes(kw)) {
        return rule.action;
      }
    }
  }

  return defaultAction;
}

let counter = 0;

/**
 * Generates a unique event ID if none provided.
 */
function defaultGenerateEventId(): string {
  counter += 1;
  return `evt-${Date.now()}-${counter}`;
}

/**
 * Pure function mapper projecting Predicate-Argument Structure (PAS) and Zero Pronoun resolutions
 * into Worldcraft @worldcraft/schema StateDeltaEvent.
 */
export function mapPASToStateDeltaEvent(
  pasRelation: PASPredicateRelation,
  zeroPronounResolutions?: ZeroPronounResolutionResult[],
  options?: EventMapperOptions
): StateDeltaEvent {
  const timestamp = options?.timestamp ?? Date.now();
  const idGen = options?.eventIdGenerator ?? defaultGenerateEventId;
  const eventId = idGen();

  const predText = pasRelation.predicateText ?? '';
  const action = normalizePredicateAction(
    predText,
    options?.customDictionary,
    options?.defaultAction
  );

  let subject: string | undefined;
  let object: string | undefined;
  let target: string | undefined;
  let isSubjectZeroPronoun = false;
  let resolvedAntecedentId: string | undefined;

  // Extract explicit arguments (ガ, ヲ, ニ)
  for (const arg of pasRelation.arguments) {
    if (arg.caseName === 'ガ' || arg.caseName === 'ガ２') {
      if (!arg.isZeroPronoun && arg.argText) {
        subject = arg.argText;
      } else if (arg.isZeroPronoun) {
        isSubjectZeroPronoun = true;
      }
    } else if (arg.caseName === 'ヲ') {
      if (!arg.isZeroPronoun && arg.argText) {
        object = arg.argText;
      }
    } else if (arg.caseName === 'ニ') {
      if (!arg.isZeroPronoun && arg.argText) {
        target = arg.argText;
      }
    }
  }

  // Handle missing/omitted subject via ZeroPronounResolver results
  if (!subject && zeroPronounResolutions) {
    const matchingRes = zeroPronounResolutions.find(
      (res) =>
        res.predicateIndex === pasRelation.predicateIndex &&
        (res.omittedCase === 'ガ' || res.omittedCase === 'ガ２')
    );

    if (matchingRes && matchingRes.bestCandidate) {
      subject = matchingRes.bestCandidate.candidate.text;
      resolvedAntecedentId = matchingRes.bestCandidate.candidate.id;
      isSubjectZeroPronoun = true;
    }
  }

  const primaryEntityId =
    subject ?? resolvedAntecedentId ?? options?.defaultEntityId ?? 'unknown_entity';

  return {
    id: eventId,
    timestamp,
    entityId: primaryEntityId,
    action,
    operation: action.toLowerCase(),
    subject,
    object,
    target,
    predicate: predText,
    payload: {
      isSubjectZeroPronoun,
      resolvedAntecedentId,
      arguments: pasRelation.arguments.map((a) => ({
        caseName: a.caseName,
        text: a.argText,
        score: a.score,
        isZeroPronoun: a.isZeroPronoun,
      })),
    },
    metadata: {
      predicateIndex: pasRelation.predicateIndex,
      source: 'PredicateArgumentEventMapper',
    },
  };
}

/**
 * Pure function batch mapper for multiple PAS relations.
 */
export function mapPASRelationsToStateDeltaEvents(
  pasRelations: PASPredicateRelation[],
  zeroPronounResolutions?: ZeroPronounResolutionResult[],
  options?: EventMapperOptions
): StateDeltaEvent[] {
  return pasRelations.map((rel) =>
    mapPASToStateDeltaEvent(rel, zeroPronounResolutions, options)
  );
}

export class PredicateArgumentEventMapper {
  private options?: EventMapperOptions;

  constructor(options?: EventMapperOptions) {
    this.options = options;
  }

  public mapRelation(
    pasRelation: PASPredicateRelation,
    zeroPronounResolutions?: ZeroPronounResolutionResult[]
  ): StateDeltaEvent {
    return mapPASToStateDeltaEvent(pasRelation, zeroPronounResolutions, this.options);
  }

  public mapRelations(
    pasRelations: PASPredicateRelation[],
    zeroPronounResolutions?: ZeroPronounResolutionResult[]
  ): StateDeltaEvent[] {
    return mapPASRelationsToStateDeltaEvents(pasRelations, zeroPronounResolutions, this.options);
  }
}
