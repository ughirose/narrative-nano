import { describe, it, expect } from 'vitest';
import {
  normalizePredicateAction,
  mapPASToStateDeltaEvent,
  mapPASRelationsToStateDeltaEvents,
  PredicateArgumentEventMapper,
  PREDICATE_ACTION_DICTIONARY,
} from '../model/PredicateArgumentEventMapper.js';
import type { PASPredicateRelation } from '../model/BiaffinePASHead.js';
import type { ZeroPronounResolutionResult } from '../model/ZeroPronounResolver.js';

describe('PredicateArgumentEventMapper', () => {
  describe('normalizePredicateAction', () => {
    it('should correctly normalize standard Japanese verbs to action codes', () => {
      expect(normalizePredicateAction('拾う')).toBe('Acquire');
      expect(normalizePredicateAction('手に入れる')).toBe('Acquire');
      expect(normalizePredicateAction('渡す')).toBe('Drop');
      expect(normalizePredicateAction('失う')).toBe('Drop');
      expect(normalizePredicateAction('向かう')).toBe('Move');
      expect(normalizePredicateAction('倒す')).toBe('ChangeState');
    });

    it('should normalize conjugated verb forms using stem keyword matching', () => {
      expect(normalizePredicateAction('拾った')).toBe('Acquire');
      expect(normalizePredicateAction('手に入れた')).toBe('Acquire');
      expect(normalizePredicateAction('手渡した')).toBe('Drop');
      expect(normalizePredicateAction('向かっている')).toBe('Move');
      expect(normalizePredicateAction('倒された')).toBe('ChangeState');
    });

    it('should respect custom dictionary overrides', () => {
      const customDict = {
        '召還する': 'Acquire' as const,
        'テレポート': 'Move' as const,
      };

      expect(normalizePredicateAction('召還する', customDict)).toBe('Acquire');
      expect(normalizePredicateAction('テレポート', customDict)).toBe('Move');
    });

    it('should fallback to default action for unknown verbs', () => {
      expect(normalizePredicateAction('考える')).toBe('ChangeState');
      expect(normalizePredicateAction('考える', undefined, 'Move')).toBe('Move');
    });
  });

  describe('mapPASToStateDeltaEvent with explicit PAS arguments', () => {
    it('should map explicit ガ, ヲ, ニ cases into subject, object, target', () => {
      const pasRelation: PASPredicateRelation = {
        predicateIndex: 5,
        predicateText: '渡す',
        arguments: [
          { caseName: 'ガ', argIndex: 0, argText: '太郎', score: 0.95, isZeroPronoun: false },
          { caseName: 'ヲ', argIndex: 2, argText: '地図', score: 0.9, isZeroPronoun: false },
          { caseName: 'ニ', argIndex: 4, argText: '花子', score: 0.88, isZeroPronoun: false },
        ],
      };

      const event = mapPASToStateDeltaEvent(pasRelation, undefined, {
        timestamp: 1700000000000,
        eventIdGenerator: () => 'evt-test-1',
      });

      expect(event.id).toBe('evt-test-1');
      expect(event.timestamp).toBe(1700000000000);
      expect(event.action).toBe('Drop');
      expect(event.subject).toBe('太郎');
      expect(event.object).toBe('地図');
      expect(event.target).toBe('花子');
      expect(event.entityId).toBe('太郎');
      expect(event.predicate).toBe('渡す');
      expect(event.payload?.isSubjectZeroPronoun).toBe(false);
    });

    it('should map Acquire action for "拾う" with subject and object', () => {
      const pasRelation: PASPredicateRelation = {
        predicateIndex: 3,
        predicateText: '拾う',
        arguments: [
          { caseName: 'ガ', argIndex: 0, argText: 'メロス', score: 0.92, isZeroPronoun: false },
          { caseName: 'ヲ', argIndex: 2, argText: '剣', score: 0.89, isZeroPronoun: false },
        ],
      };

      const event = mapPASToStateDeltaEvent(pasRelation);

      expect(event.action).toBe('Acquire');
      expect(event.subject).toBe('メロス');
      expect(event.object).toBe('剣');
      expect(event.target).toBeUndefined();
      expect(event.entityId).toBe('メロス');
    });

    it('should map Move action for "向かう" with target location', () => {
      const pasRelation: PASPredicateRelation = {
        predicateIndex: 4,
        predicateText: '向かう',
        arguments: [
          { caseName: 'ガ', argIndex: 0, argText: 'セリヌンティウス', score: 0.95, isZeroPronoun: false },
          { caseName: 'ニ', argIndex: 2, argText: '王宮', score: 0.87, isZeroPronoun: false },
        ],
      };

      const event = mapPASToStateDeltaEvent(pasRelation);

      expect(event.action).toBe('Move');
      expect(event.subject).toBe('セリヌンティウス');
      expect(event.target).toBe('王宮');
      expect(event.object).toBeUndefined();
    });
  });

  describe('ZeroPronounResolver integration for omitted subject', () => {
    it('should bind antecedent candidate entity when subject (ガ) is omitted (zero pronoun)', () => {
      const pasRelation: PASPredicateRelation = {
        predicateIndex: 2,
        predicateText: '拾い上げる',
        arguments: [
          { caseName: 'ガ', argIndex: null, score: 0.85, isZeroPronoun: true },
          { caseName: 'ヲ', argIndex: 1, argText: '落ちていた鍵', score: 0.91, isZeroPronoun: false },
        ],
      };

      const zeroPronounResolutions: ZeroPronounResolutionResult[] = [
        {
          predicateIndex: 2,
          predicateText: '拾い上げる',
          omittedCase: 'ガ',
          zeroPronounScore: 0.85,
          isResolved: true,
          bestCandidate: {
            candidate: {
              id: 'char-melos',
              text: 'メロス',
              entityType: 'character',
              sentenceDistance: 1,
            },
            totalScore: 0.88,
            anaphoraLikelihood: 0.75,
            breakdown: {
              pasScore: 0.85,
              distanceScore: 0.75,
              syntacticProminence: 0.9,
              salienceScore: 1.2,
            },
          },
          rankedCandidates: [],
        },
      ];

      const event = mapPASToStateDeltaEvent(pasRelation, zeroPronounResolutions);

      expect(event.action).toBe('Acquire');
      expect(event.subject).toBe('メロス');
      expect(event.object).toBe('落ちていた鍵');
      expect(event.entityId).toBe('メロス');
      expect(event.payload?.isSubjectZeroPronoun).toBe(true);
      expect(event.payload?.resolvedAntecedentId).toBe('char-melos');
    });
  });

  describe('Batch mapping and Mapper Instance', () => {
    it('should map multiple PAS relations using mapPASRelationsToStateDeltaEvents', () => {
      const pasRelations: PASPredicateRelation[] = [
        {
          predicateIndex: 2,
          predicateText: '手に入れる',
          arguments: [
            { caseName: 'ガ', argIndex: 0, argText: '勇者', score: 0.9, isZeroPronoun: false },
            { caseName: 'ヲ', argIndex: 1, argText: '宝箱', score: 0.88, isZeroPronoun: false },
          ],
        },
        {
          predicateIndex: 6,
          predicateText: '倒す',
          arguments: [
            { caseName: 'ガ', argIndex: 4, argText: '勇者', score: 0.93, isZeroPronoun: false },
            { caseName: 'ヲ', argIndex: 5, argText: '魔王', score: 0.91, isZeroPronoun: false },
          ],
        },
      ];

      const events = mapPASRelationsToStateDeltaEvents(pasRelations);

      expect(events).toHaveLength(2);
      expect(events[0].action).toBe('Acquire');
      expect(events[0].object).toBe('宝箱');
      expect(events[1].action).toBe('ChangeState');
      expect(events[1].object).toBe('魔王');
    });

    it('should support PredicateArgumentEventMapper class instance with default options', () => {
      const mapper = new PredicateArgumentEventMapper({
        defaultEntityId: 'hero-1',
        defaultAction: 'ChangeState',
      });

      const pasRelation: PASPredicateRelation = {
        predicateIndex: 1,
        predicateText: '討伐する',
        arguments: [
          { caseName: 'ヲ', argIndex: 0, argText: 'ドラゴン', score: 0.8, isZeroPronoun: false },
        ],
      };

      const event = mapper.mapRelation(pasRelation);

      expect(event.entityId).toBe('hero-1');
      expect(event.object).toBe('ドラゴン');
      expect(event.action).toBe('ChangeState');
    });
  });
});
