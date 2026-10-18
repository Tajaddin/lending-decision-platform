# How this project is run

Scrum, scaled to one developer:

- **Backlog** as data (`docs/agile/backlog.json`), synced to GitHub issues with labels for type, area and story
  points (`scripts/agile/sync_backlog.py`). Every story has acceptance criteria.
- **Sprints** are GitHub milestones with a goal. Work happens on a `sprint-N` branch; commits reference their
  issue (`(#6)`); the sprint ends with one pull request whose description closes the sprint's issues, merged
  only when CI is green.
- **Definition of done**: tests for the behaviour, CI green, numbers in docs traced to a committed result file,
  secret scan clean.
- **Retro** at the end of each sprint, committed here: what was planned, what shipped, the measured results, what
  went well, what did not, and the actions carried into the next sprint.
- Work that needs an account the team does not have yet stays in the **Backlog** milestone, labelled
  `blocked:account`.
