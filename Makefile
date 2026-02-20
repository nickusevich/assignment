.PHONY: setup format lint typecheck test check all clean

# Install dependencies with UV
setup:
	uv pip install -e "."
	@echo "✅ Setup complete!"

# Format code
format:
	@echo "🎨 Formatting code..."
	black src/
	@echo "✅ Formatting complete!"

# Lint code
lint:
	@echo "🔍 Linting code..."
	ruff check src/ --fix
	@echo "✅ Linting complete!"

# Type check
typecheck:
	@echo "🔎 Type checking..."
	mypy src/
	@echo "✅ Type checking complete!"

# # Run tests
# test:
# 	@echo "🧪 Running tests..."
# 	pytest
# 	@echo "✅ Tests complete!"

# Run all checks (before git push!)
check: format lint typecheck
	@echo ""
	@echo "═══════════════════════════════════"
	@echo "  ✅ All checks passed!"
	@echo "  Ready to commit and push!"
	@echo "═══════════════════════════════════"

# Clean cache
clean:
	@echo "🧹 Cleaning cache..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Clean complete!"

# Full pipeline
all: setup check test
	@echo "🎉 Everything is ready!"