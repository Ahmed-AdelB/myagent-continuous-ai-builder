# Tri-Agent Pull Request

## 📋 Summary
<!-- Brief description of what this PR does -->



## 🎯 Related Issue(s)
<!-- Link to related GitHub issues -->
Closes #
Related to #

## 📝 Type of Change
<!-- Mark the relevant option with an 'x' -->
- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🔧 Configuration/Infrastructure change
- [ ] ⚡ Performance improvement
- [ ] ♻️ Refactoring (no functional changes)
- [ ] ✅ Test additions/improvements

## 🔍 Changes Made
<!-- Detailed list of changes -->
-
-
-

## 🧪 Testing Performed
<!-- Describe the tests you ran -->
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed
- [ ] All existing tests passing

**Test Coverage**: `___%` (target: ≥90%)

```bash
# Commands used for testing
pytest tests/ -v --cov
```

## 📊 Validation Results

### Layer 1: Self-Check (Weight: 0.2)
<!-- Implementing agent's self-assessment -->
**Agent**: <!-- Claude | Codex | Gemini -->
**Score**: `___/1.0`
**Notes**:
-

### Layer 2: Peer Review (Weight: 0.3)
<!-- Review by different agent -->
**Reviewer Agent**: <!-- Claude | Codex | Gemini (different from implementing agent) -->
**Score**: `___/1.0`
**Notes**:
-

### Layer 3: Automated Tests (Weight: 0.3)
**Test Results**: <!-- PASS | FAIL -->
**Coverage**: `___%`
**Score**: `___/1.0`
**Failed Tests** (if any):
-

### Layer 4: Static Analysis (Weight: 0.2)
**Tools Run**:
- [ ] ruff (formatting & linting)
- [ ] mypy (type checking)
- [ ] bandit (security)
- [ ] pylint (code quality)

**Score**: `___/1.0`
**Issues Found**:
-

### 🎯 Overall Validation Score
**Formula**: `(0.2 × self) + (0.3 × peer) + (0.3 × tests) + (0.2 × static)`

**Score**: `___/1.0`

**Quality Gate Status**:
- [ ] ✅ Excellent (0.95-1.0) - Ready to merge
- [ ] ✅ Good (0.85-0.94) - Ready to merge with minor notes
- [ ] ⚠️ Acceptable (0.70-0.84) - Merge with caution, create follow-up issues
- [ ] ❌ Needs Revision (<0.70) - DO NOT MERGE

## 🤖 Tri-Agent Approval

### Required: 3/3 Unanimous Approval for Critical Changes

**Agent Votes**:
- [ ] ✅ **Claude** (Sonnet 4.5): `APPROVE | REQUEST_CHANGES | COMMENT`
  - Reasoning:

- [ ] ✅ **Codex** (GPT-5.1): `APPROVE | REQUEST_CHANGES | COMMENT`
  - Reasoning:

- [ ] ✅ **Gemini** (2.5/3.0 Pro): `APPROVE | REQUEST_CHANGES | COMMENT`
  - Reasoning:

**Consensus Status**: `___/3 APPROVE`

**Conflict Resolution** (if split decision):
- Lead Agent: <!-- Based on domain: architecture→Claude, implementation→Codex, security→Gemini -->
- Resolution:

## 🔒 Security Checklist
- [ ] No secrets/credentials in code
- [ ] No eval() or exec() calls
- [ ] Input validation implemented
- [ ] SQL injection prevented (parameterized queries)
- [ ] XSS prevention measures
- [ ] CSRF protection where applicable
- [ ] Dependencies scanned for vulnerabilities
- [ ] SBOM updated

## 📦 Deployment Notes
<!-- Any special deployment considerations -->
- [ ] Database migrations required
- [ ] Environment variables changed
- [ ] Configuration updates needed
- [ ] Feature flags configured
- [ ] Rollback plan documented

**Rollback Plan**:
<!-- How to revert this change if needed -->

## 📸 Screenshots/Videos
<!-- If applicable, add screenshots or videos -->


## 🔗 Additional Context
<!-- Any other context, links to documentation, etc. -->


---

## ✅ Pre-Merge Checklist

**Implementing Agent**:
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated (if needed)
- [ ] Tests added/updated
- [ ] All tests passing locally
- [ ] No new warnings introduced
- [ ] Commit messages follow convention

**Reviewer Agents**:
- [ ] Code review completed
- [ ] Logic verified
- [ ] Edge cases considered
- [ ] Performance implications assessed
- [ ] Security review completed

**Maintainer**:
- [ ] Validation score ≥ 0.85
- [ ] 3/3 agent approval (for critical changes)
- [ ] CI/CD pipeline passing
- [ ] Conflicts resolved
- [ ] Branch up to date with main

---

## 🏷️ Git Commit Format

```
<type>(<scope>): <subject>

<body>

🤖 Tri-Agent Approval:
✅ Claude Code (Sonnet 4.5): APPROVE
✅ Codex (GPT-5.1): APPROVE
✅ Gemini (2.5/3.0 Pro): APPROVE

Validation Score: X.XX/1.0 (Excellent|Good|Acceptable)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Codex <noreply@openai.com>
Co-Authored-By: Gemini <noreply@google.com>
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`, `ci`

---

**Note**: This PR template enforces the Tri-Agent SDLC validation framework with 4-layer validation (self-check + peer review + automated tests + static analysis) and requires unanimous approval from all 3 agents for critical changes.
