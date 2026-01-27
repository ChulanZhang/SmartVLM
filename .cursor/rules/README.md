# Cursor Rules

This directory contains Cursor AI rule files that provide project-specific context and guidance for the AI assistant.

## Rule Files

### Core Rules

1. **research-code-standards.mdc** (alwaysApply: true)
   - High-quality research code standards
   - Code organization and naming conventions
   - Error handling and documentation practices

2. **python-standards.mdc** (globs: **/*.py)
   - PEP 8, import order, Google-style docstrings

3. **pytorch-best-practices.mdc** (globs: **/*.py)
   - PyTorch tensor operations and device management
   - Model architecture patterns
   - Performance optimization guidelines

4. **ada-llava-patterns.mdc** (globs: src/adallava/**/*.py)
   - AdaLLaVA-specific architecture patterns
   - Adaptive computation design principles
   - Latency-aware scheduling patterns

5. **experiment-reproducibility.mdc** (globs: train/eval scripts)
   - Random seed management
   - Configuration management
   - Logging and checkpointing best practices

## Rule File Format

Rule files use `.mdc` format with YAML frontmatter and Markdown content:

```markdown
---
description: Brief description
globs: **/*.py  # File patterns (optional)
alwaysApply: false  # Apply to all conversations (optional)
---

# Rule Content
...
```

## Rule Types

- **Always Apply** (`alwaysApply: true`): Universal rules for all conversations
- **File-Specific** (`globs: **/*.py`): Rules that apply when working with matching files
