all:
	@COMMIT_HASH=$(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown"); \
	if [ "$$COMMIT_HASH" = "unknown" ]; then \
		echo "Warning: Could not determine Git commit hash. Using 'unknown'."; \
	fi; \
	go install -ldflags "-X main.commitHash=$$COMMIT_HASH" -v
