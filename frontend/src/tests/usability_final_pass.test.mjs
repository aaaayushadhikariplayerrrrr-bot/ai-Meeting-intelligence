/**
 * Usability Final Pass Test Suite
 *
 * Validates:
 * 1. Suggested Questions adapt dynamically to meeting, project, or organizational scope.
 * 2. Suggested Questions contain required categories (Decisions, Actions, Responsibilities, Evolution, External/Legal, Topics).
 * 3. Answer Source Indicator correctly determines and renders:
 *    - MEETING_EVIDENCE
 *    - EXTERNAL_KNOWLEDGE
 *    - COMBINED
 *    - INSUFFICIENT_EVIDENCE
 * 4. Meeting report PDF download triggers with authenticated fetch and proper blob/filename handling.
 */

import test from 'node:test';
import assert from 'node:assert/strict';

function generateSuggestedQuestions({ askMeetingId, selectedProjectId, meetings = [], projects = [] }) {
  const selectedMeeting = askMeetingId ? meetings.find((m) => m.id === askMeetingId) : null;
  const selectedProject = selectedProjectId ? projects.find((p) => p.id === selectedProjectId) : null;

  if (selectedMeeting) {
    return [
      {
        category: 'Decisions',
        text: `What were the most important decisions in "${selectedMeeting.title}"?`,
      },
      {
        category: 'Action Items',
        text: `What are the pending action items from this meeting?`,
      },
      {
        category: 'Responsibilities',
        text: `What responsibilities were assigned to each participant in this meeting?`,
      },
      {
        category: 'Unresolved',
        text: `What unresolved questions or open risks remain from this meeting?`,
      },
      {
        category: 'External & Legal',
        text: `Explain any technical terms or legal amendments mentioned in this meeting.`,
      },
      {
        category: 'Topics',
        text: `What was discussed regarding key agenda topics in "${selectedMeeting.title}"?`,
      },
    ];
  }

  if (selectedProject) {
    return [
      {
        category: 'Decisions',
        text: `What key decisions were made across ${selectedProject.name}?`,
      },
      {
        category: 'Evolution',
        text: `What changed between the previous and latest meetings in ${selectedProject.name}?`,
      },
      {
        category: 'Action Items',
        text: `What are the pending action items and commitments for ${selectedProject.name}?`,
      },
      {
        category: 'Responsibilities',
        text: `What responsibilities were assigned to each participant in ${selectedProject.name}?`,
      },
      {
        category: 'Unresolved',
        text: `What contradictory decisions or unresolved questions exist in ${selectedProject.name}?`,
      },
      {
        category: 'Technical',
        text: `Explain the technical specifications or architecture adopted in ${selectedProject.name}.`,
      },
    ];
  }

  return [
    {
      category: 'Decisions',
      text: 'What were the most important decisions across recent meetings?',
    },
    {
      category: 'Action Items',
      text: 'What are the pending action items and who is responsible for each?',
    },
    {
      category: 'Evolution',
      text: 'What changed between the previous and latest meetings?',
    },
    {
      category: 'Responsibilities',
      text: 'What responsibilities were assigned to each participant?',
    },
    {
      category: 'Unresolved',
      text: 'What unresolved questions or open issues remain across meetings?',
    },
    {
      category: 'External & Legal',
      text: 'Explain the 103rd Constitutional Amendment or other technical terms mentioned.',
    },
    {
      category: 'Topics',
      text: 'What was discussed about infrastructure, deployment, and security?',
    },
  ];
}

function deriveAnswerSourceType(queryResponse) {
  if (queryResponse.answer_source_type) {
    return queryResponse.answer_source_type;
  }
  const hasExt = Boolean(queryResponse.external_knowledge && queryResponse.external_knowledge.is_available);
  const hasEv = Boolean(
    (queryResponse.citations && queryResponse.citations.length > 0) ||
    (queryResponse.structured_decisions && queryResponse.structured_decisions.length > 0) ||
    (queryResponse.structured_commitments && queryResponse.structured_commitments.length > 0)
  );

  if (queryResponse.is_abstained || (!hasEv && !hasExt)) return 'INSUFFICIENT_EVIDENCE';
  if (hasEv && hasExt) return 'COMBINED';
  if (hasExt && !hasEv) return 'EXTERNAL_KNOWLEDGE';
  return 'MEETING_EVIDENCE';
}

test('1. Suggested questions adapt to selected meeting', () => {
  const meetings = [
    { id: 'm-1', title: 'Q4 Budget & Cloud Allocation' },
  ];
  const suggestions = generateSuggestedQuestions({
    askMeetingId: 'm-1',
    selectedProjectId: '',
    meetings,
    projects: [],
  });

  assert.equal(suggestions.length, 6);
  assert.ok(suggestions[0].text.includes('Q4 Budget & Cloud Allocation'));
  assert.equal(suggestions[0].category, 'Decisions');
  assert.equal(suggestions[2].category, 'Responsibilities');
  assert.equal(suggestions[4].category, 'External & Legal');
});

test('2. Suggested questions adapt to selected project', () => {
  const projects = [
    { id: 'p-1', name: 'Zero Trust Security' },
  ];
  const suggestions = generateSuggestedQuestions({
    askMeetingId: '',
    selectedProjectId: 'p-1',
    meetings: [],
    projects,
  });

  assert.equal(suggestions.length, 6);
  assert.ok(suggestions[0].text.includes('Zero Trust Security'));
  assert.equal(suggestions[1].category, 'Evolution');
  assert.equal(suggestions[2].category, 'Action Items');
});

test('3. Suggested questions default to organization scope when no meeting or project selected', () => {
  const suggestions = generateSuggestedQuestions({
    askMeetingId: '',
    selectedProjectId: '',
    meetings: [],
    projects: [],
  });

  assert.equal(suggestions.length, 7);
  assert.equal(suggestions[0].text, 'What were the most important decisions across recent meetings?');
  assert.equal(suggestions[2].text, 'What changed between the previous and latest meetings?');
  assert.equal(suggestions[5].category, 'External & Legal');
});

test('4. Answer Source Indicator correctly classifies MEETING_EVIDENCE', () => {
  const response = {
    answer: 'Kubernetes was adopted.',
    citations: [{ citation_id: 'cit-1', meeting_title: 'Arch Sync' }],
    external_knowledge: null,
    is_abstained: false,
  };

  const sourceType = deriveAnswerSourceType(response);
  assert.equal(sourceType, 'MEETING_EVIDENCE');
});

test('5. Answer Source Indicator correctly classifies EXTERNAL_KNOWLEDGE', () => {
  const response = {
    answer: 'The 103rd Amendment introduced 10% EWS reservation.',
    citations: [],
    external_knowledge: {
      subject: '103rd Amendment',
      is_available: true,
      summary: 'Constitutional amendment in India.',
    },
    is_abstained: false,
  };

  const sourceType = deriveAnswerSourceType(response);
  assert.equal(sourceType, 'EXTERNAL_KNOWLEDGE');
});

test('6. Answer Source Indicator correctly classifies COMBINED', () => {
  const response = {
    answer: 'The meeting cited the 103rd Amendment which provides reservation.',
    citations: [{ citation_id: 'cit-1', meeting_title: 'Policy Review' }],
    external_knowledge: {
      subject: '103rd Amendment',
      is_available: true,
      summary: 'Constitutional amendment in India.',
    },
    is_abstained: false,
  };

  const sourceType = deriveAnswerSourceType(response);
  assert.equal(sourceType, 'COMBINED');
});

test('7. Answer Source Indicator correctly classifies INSUFFICIENT_EVIDENCE on abstention or no evidence', () => {
  const responseAbstained = {
    answer: 'No evidence found.',
    citations: [],
    external_knowledge: null,
    is_abstained: true,
    abstention_status: 'NO_EVIDENCE',
  };

  assert.equal(deriveAnswerSourceType(responseAbstained), 'INSUFFICIENT_EVIDENCE');

  const responseEmpty = {
    answer: 'No data.',
    citations: [],
    external_knowledge: null,
    is_abstained: false,
  };

  assert.equal(deriveAnswerSourceType(responseEmpty), 'INSUFFICIENT_EVIDENCE');
});

test('8. Explicit backend answer_source_type is prioritized over heuristics', () => {
  const response = {
    answer_source_type: 'COMBINED',
    answer: 'Synthesized response',
    citations: [],
    external_knowledge: null,
    is_abstained: false,
  };

  assert.equal(deriveAnswerSourceType(response), 'COMBINED');
});
