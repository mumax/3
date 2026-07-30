#!/bin/bash

# Bootstrap mumax3-for-mac on a fresh Apple-silicon Mac.
# Compatible with the Bash 3.2 shipped by macOS.

set -Eeuo pipefail

PROGRAM_NAME="mumax3-for-mac installer"
REPOSITORY_URL="${MUMAX3_REPOSITORY_URL:-https://github.com/TaewoooPark/mumax3-for-mac.git}"
MINIMUM_MACOS_MAJOR=14
MINIMUM_GO_VERSION="1.22.4"
CURRENT_STEP="initialization"
NO_PROFILE=0
SOURCE_DIR="${MUMAX3_SOURCE_DIR:-}"
USER_HOME_DIR="${HOME:?HOME is not set}"
export PATH="/opt/homebrew/bin:/usr/local/go/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

log() {
	printf '\n==> %s\n' "$1"
}

note() {
	printf '    %s\n' "$1"
}

warn() {
	printf 'WARNING: %s\n' "$1" >&2
}

fail() {
	printf 'ERROR: %s\n' "$1" >&2
	exit 1
}

on_error() {
	status=$?
	trap - ERR
	printf '\nERROR: %s failed during: %s\n' "$PROGRAM_NAME" "$CURRENT_STEP" >&2
	printf '       Exit status: %s\n' "$status" >&2
	printf '       Fix the reported error and run the installer again; completed steps are reused.\n' >&2
	exit "$status"
}

on_interrupt() {
	trap - INT TERM
	printf '\nERROR: Installation interrupted during: %s\n' "$CURRENT_STEP" >&2
	exit 130
}

trap on_error ERR
trap on_interrupt INT TERM

usage() {
	cat <<'EOF'
Usage: install-macos.sh [options]

Install and verify mumax3-for-mac on Apple Silicon.

Options:
  --source-dir DIR  Clone into or build from DIR.
                    Default: ~/mumax3-for-mac
  --no-profile      Do not add Homebrew or mumax3 to ~/.zprofile.
  -h, --help        Show this help.

Environment overrides:
  MUMAX3_SOURCE_DIR       Same as --source-dir.
  MUMAX3_REPOSITORY_URL   Repository to clone.

The installer may open Apple's Command Line Tools installer and Homebrew may
ask for an administrator password. Existing source directories and shell
configuration are not overwritten.
EOF
}

while (($# > 0)); do
	case "$1" in
		--source-dir)
			(($# >= 2)) || fail "--source-dir requires a directory"
			SOURCE_DIR=$2
			shift 2
			;;
		--no-profile)
			NO_PROFILE=1
			shift
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			fail "unknown option: $1"
			;;
	esac
done

version_at_least() {
	/usr/bin/awk -v actual="$1" -v required="$2" '
		BEGIN {
			split(actual, a, ".")
			split(required, r, ".")
			for (i = 1; i <= 3; i++) {
				av = (a[i] == "" ? 0 : a[i]) + 0
				rv = (r[i] == "" ? 0 : r[i]) + 0
				if (av > rv) {
					exit 0
				}
				if (av < rv) {
					exit 1
				}
			}
			exit 0
		}
	'
}

resolve_profile_path() {
	if [[ -n "${ZDOTDIR:-}" ]]; then
		case "$ZDOTDIR" in
			/*) printf '%s/.zprofile\n' "${ZDOTDIR%/}" ;;
			*) printf '%s/%s/.zprofile\n' "${USER_HOME_DIR%/}" "${ZDOTDIR%/}" ;;
		esac
	else
		printf '%s/.zprofile\n' "${USER_HOME_DIR%/}"
	fi
}

ensure_profile_line() {
	line=$1
	description=$2

	if ((NO_PROFILE == 1)); then
		return
	fi

	profile_path=$(resolve_profile_path)
	profile_dir=$(/usr/bin/dirname "$profile_path")
	/bin/mkdir -p "$profile_dir"
	[[ -e "$profile_path" ]] || /usr/bin/touch "$profile_path"
	if ! /usr/bin/grep -Fqx "$line" "$profile_path"; then
		{
			printf '\n# Added by the mumax3-for-mac installer: %s\n' "$description"
			printf '%s\n' "$line"
		} >>"$profile_path"
		note "Updated $profile_path ($description)"
	fi
}

developer_tools_ready() {
	/usr/bin/xcrun --find clang >/dev/null 2>&1 &&
		command -v git >/dev/null 2>&1 &&
		command -v make >/dev/null 2>&1
}

ensure_developer_tools() {
	if developer_tools_ready; then
		note "Apple Command Line Tools are ready."
		return
	fi

	if /usr/bin/xcode-select -p >/dev/null 2>&1; then
		fail "developer tools are selected but clang, git, or make is unavailable. Run 'xcode-select --install' or repair the active Xcode installation."
	fi

	CURRENT_STEP="starting the Apple Command Line Tools installer"
	log "Apple Command Line Tools are required"
	note "A macOS installer window will open. Complete it to continue."
	/usr/bin/xcode-select --install

	CURRENT_STEP="waiting for Apple Command Line Tools"
	elapsed=0
	while ! developer_tools_ready; do
		if ((elapsed >= 7200)); then
			fail "timed out waiting for Command Line Tools after 120 minutes"
		fi
		/bin/sleep 5
		elapsed=$((elapsed + 5))
		if ((elapsed % 30 == 0)); then
			note "Still waiting for the Command Line Tools installer (${elapsed}s). Press Ctrl-C to cancel."
		fi
	done
	note "Apple Command Line Tools installation completed."
}

go_version_is_supported() {
	command -v go >/dev/null 2>&1 || return 1

	go_version_raw=$(go env GOVERSION 2>/dev/null) || return 1
	go_version=${go_version_raw#go}
	version_at_least "$go_version" "$MINIMUM_GO_VERSION"
}

go_host_is_native() {
	[[ "$(go env GOHOSTOS 2>/dev/null)" == "darwin" ]] &&
		[[ "$(go env GOHOSTARCH 2>/dev/null)" == "arm64" ]]
}

go_target_is_native() {
	[[ "$(go env GOOS 2>/dev/null)" == "darwin" ]] &&
		[[ "$(go env GOARCH 2>/dev/null)" == "arm64" ]]
}

print_go_environment() {
	for go_key in GOVERSION GOHOSTOS GOHOSTARCH GOOS GOARCH CGO_ENABLED; do
		go_value=$(go env "$go_key" 2>/dev/null || printf '<unavailable>')
		printf '%s=%s\n' "$go_key" "$go_value"
	done
}

go_is_compatible() {
	go_version_is_supported && go_host_is_native && go_target_is_native
}

ensure_native_homebrew() {
	if [[ ! -x /opt/homebrew/bin/brew ]]; then
		CURRENT_STEP="installing native Apple-silicon Homebrew"
		log "Installing Homebrew"
		note "Homebrew may request your administrator password."
		homebrew_installer=$(
			/usr/bin/curl -fsSL \
				https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh
		)
		/bin/bash -c "$homebrew_installer"
	fi

	[[ -x /opt/homebrew/bin/brew ]] ||
		fail "native Homebrew was not installed at /opt/homebrew"

	eval "$(/opt/homebrew/bin/brew shellenv)"
	ensure_profile_line \
		'eval "$(/opt/homebrew/bin/brew shellenv)"' \
		"enable native Homebrew"
}

ensure_go() {
	if command -v go >/dev/null 2>&1 &&
		go_host_is_native &&
		! go_target_is_native; then
		print_go_environment >&2
		fail "the native Go installation is being overridden. Unset GOOS and GOARCH, then retry."
	fi

	if go_is_compatible; then
		note "Compatible Go found: $(go version)"
		return
	fi

	CURRENT_STEP="installing Go ${MINIMUM_GO_VERSION} or newer"
	log "Installing Go"
	if command -v go >/dev/null 2>&1; then
		warn "The existing Go installation is too old or is not targeting darwin/arm64."
	fi

	ensure_native_homebrew
	CURRENT_STEP="installing Go ${MINIMUM_GO_VERSION} or newer"
	if /opt/homebrew/bin/brew list --versions go >/dev/null 2>&1; then
		/opt/homebrew/bin/brew upgrade go
	else
		/opt/homebrew/bin/brew install go
	fi
	hash -r

	if ! go_is_compatible; then
		print_go_environment >&2
		fail "Go must be ${MINIMUM_GO_VERSION}+ and target native darwin/arm64."
	fi
	note "Installed $(go version)"
}

is_metal_source_tree() {
	candidate_dir=$1
	[[ -f "$candidate_dir/go.mod" ]] &&
		/usr/bin/grep -Eq '^module[[:space:]]+github\.com/mumax/3$' \
			"$candidate_dir/go.mod" &&
		[[ -s "$candidate_dir/cuda/metal/kernels/mumax3_kernels.metal" ]]
}

detect_local_checkout() {
	script_path="${BASH_SOURCE[0]:-}"
	[[ -n "$script_path" && -f "$script_path" ]] || return 1

	script_dir=$(
		CDPATH= cd -- "$(/usr/bin/dirname "$script_path")" &&
			/bin/pwd
	)
	is_metal_source_tree "$script_dir" || return 1
	printf '%s\n' "$script_dir"
}

prepare_source() {
	if [[ -z "$SOURCE_DIR" ]]; then
		if local_checkout=$(detect_local_checkout); then
			SOURCE_DIR=$local_checkout
		else
			SOURCE_DIR="${USER_HOME_DIR%/}/mumax3-for-mac"
		fi
	fi

	case "$SOURCE_DIR" in
		/*) ;;
		*) SOURCE_DIR="$PWD/$SOURCE_DIR" ;;
	esac

	if [[ ! -e "$SOURCE_DIR" ]]; then
		CURRENT_STEP="cloning mumax3-for-mac"
		log "Cloning mumax3-for-mac"
		/usr/bin/git clone --depth 1 "$REPOSITORY_URL" "$SOURCE_DIR"
	elif is_metal_source_tree "$SOURCE_DIR"; then
		note "Using existing source directory: $SOURCE_DIR"
	else
		fail "$SOURCE_DIR already exists and is not a mumax3-for-mac source tree. Choose another location with --source-dir."
	fi

	is_metal_source_tree "$SOURCE_DIR" ||
		fail "the source tree at $SOURCE_DIR does not contain the Metal backend"
}

configure_binary_path() {
	go_bin_dir=$(go env GOBIN)
	if [[ -z "$go_bin_dir" ]]; then
		go_bin_dir="$(go env GOPATH)/bin"
	fi
	MUMAX3_BINARY="${go_bin_dir%/}/mumax3"

	profile_path_line="export PATH=\"${go_bin_dir%/}:\$PATH\""
	ensure_profile_line "$profile_path_line" "make mumax3 available on PATH"
	export PATH="${go_bin_dir%/}:$PATH"
	hash -r
}

main() {
	CURRENT_STEP="checking macOS compatibility"
	log "Checking this Mac"

	machine_arch=$(/usr/bin/uname -m)
	if [[ "$machine_arch" != "arm64" ]]; then
		if [[ "$machine_arch" == "x86_64" ]] &&
			[[ "$(/usr/sbin/sysctl -n hw.optional.arm64 2>/dev/null || true)" == "1" ]]; then
			fail "Terminal is running through Rosetta. Open a native Apple-silicon Terminal and retry."
		fi
		fail "unsupported architecture '$machine_arch'; the Metal backend requires Apple Silicon"
	fi

	macos_version=$(/usr/bin/sw_vers -productVersion)
	macos_major=${macos_version%%.*}
	case "$macos_major" in
		''|*[!0-9]*) fail "could not parse macOS version '$macos_version'" ;;
	esac
	((macos_major >= MINIMUM_MACOS_MAJOR)) ||
		fail "macOS ${MINIMUM_MACOS_MAJOR} or newer is required; found $macos_version"
	note "Apple Silicon, macOS $macos_version"

	CURRENT_STEP="checking Apple Command Line Tools"
	ensure_developer_tools

	CURRENT_STEP="checking Go"
	ensure_go

	if [[ "$(go env CGO_ENABLED)" != "1" ]]; then
		fail "CGO is disabled. Unset CGO_ENABLED or set CGO_ENABLED=1 and retry."
	fi

	CURRENT_STEP="preparing the source tree"
	prepare_source

	CURRENT_STEP="building the Metal backend"
	log "Building mumax3 with Metal"
	(
		cd "$SOURCE_DIR"
		MACOSX_DEPLOYMENT_TARGET=14.0 CGO_ENABLED=1 make
	)

	CURRENT_STEP="locating the installed executable"
	configure_binary_path
	[[ -x "$MUMAX3_BINARY" ]] ||
		fail "build completed but no executable was found at $MUMAX3_BINARY"

	CURRENT_STEP="running the Metal smoke test"
	log "Verifying the Metal backend"
	"$MUMAX3_BINARY" -test

	CURRENT_STEP="complete"
	log "Installation complete"
	note "Source: $SOURCE_DIR"
	note "Executable: $MUMAX3_BINARY"
	if ((NO_PROFILE == 0)); then
		note "Open a new Terminal (or run: source \"$profile_path\") before running: mumax3 -test"
	else
		note "For this shell, add: export PATH=\"$(/usr/bin/dirname "$MUMAX3_BINARY"):\$PATH\""
	fi
}

main
