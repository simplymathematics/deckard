#!/bin/bash
# DVC Cache Management Utility
# Usage: bash scripts/manage_dvc_cache.sh [command]

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DVC_CACHE_DIR="$REPO_ROOT/.dvc/cache"
DOCS_NOTEBOOKS_DIR="$REPO_ROOT/docs/notebooks"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Show cache status
show_status() {
    print_header "DVC Cache Status"
    
    if [ ! -d "$DVC_CACHE_DIR" ]; then
        print_warning "Cache directory not found: $DVC_CACHE_DIR"
        echo "Run 'make html' in docs/ to initialize cache"
        return
    fi
    
    # Cache size
    cache_size=$(du -sh "$DVC_CACHE_DIR" 2>/dev/null | cut -f1)
    echo "Cache location: $DVC_CACHE_DIR"
    echo "Cache size: $cache_size"
    
    # Number of cached files
    cache_files=$(find "$DVC_CACHE_DIR" -type f 2>/dev/null | wc -l)
    echo "Cached files: $cache_files"
    
    # Pipeline status
    echo ""
    print_header "Pipeline Status"
    cd "$DOCS_NOTEBOOKS_DIR"
    dvc status || print_warning "No stages to report"
    
    # Check for uncommitted changes
    echo ""
    print_header "DVC Lock Status"
    if [ -f "$DOCS_NOTEBOOKS_DIR/dvc.lock" ]; then
        echo "dvc.lock exists and is tracked"
        changes=$(git diff --name-only "$DOCS_NOTEBOOKS_DIR/dvc.lock" 2>/dev/null | wc -l)
        if [ "$changes" -gt 0 ]; then
            print_warning "dvc.lock has uncommitted changes"
        else
            print_success "dvc.lock is clean"
        fi
    else
        print_warning "dvc.lock not found"
    fi
}

# Clear cache (forces rebuild)
clear_cache() {
    print_header "Clearing DVC Cache"
    
    if [ -d "$DVC_CACHE_DIR" ]; then
        read -p "Remove $DVC_CACHE_DIR? This will force a full rebuild. (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$DVC_CACHE_DIR"
            print_success "Cache cleared"
            echo "Next 'make html' will rebuild all notebooks from scratch"
        else
            print_warning "Cache clear cancelled"
        fi
    else
        print_warning "Cache directory not found"
    fi
}

# Rebuild notebooks
rebuild() {
    print_header "Rebuilding Notebooks"
    echo "Executing: cd $DOCS_NOTEBOOKS_DIR && make html DVC_REPRO_ARGS=\"--force\""
    cd "$DOCS_NOTEBOOKS_DIR"
    make html DVC_REPRO_ARGS="--force"
    print_success "Rebuild complete"
}

# Rebuild only changed stages
rebuild_changed() {
    print_header "Rebuilding Changed Stages Only"
    cd "$DOCS_NOTEBOOKS_DIR"
    make notebooks
    print_success "Rebuild complete"
}

# List cached stages
list_cached() {
    print_header "Cached Notebook Stages"
    cd "$DOCS_NOTEBOOKS_DIR"
    
    if [ ! -f "dvc.lock" ]; then
        print_error "dvc.lock not found"
        return 1
    fi
    
    # Extract stage names from dvc.lock
    grep "^  notebook_" dvc.lock | sed 's/:$//' | sed 's/^  /  - /'
    
    echo ""
    echo "Run 'make notebooks DVC_STAGE=<stage_name>' to rebuild a specific stage"
}

# Usage
show_usage() {
    echo "DVC Cache Management Utility"
    echo ""
    echo "Usage: bash scripts/manage_dvc_cache.sh [command]"
    echo ""
    echo "Commands:"
    echo "  status              Show cache status and pipeline state"
    echo "  clear               Clear cache (forces full rebuild on next build)"
    echo "  list                List cached notebook stages"
    echo "  rebuild             Force rebuild all notebooks"
    echo "  rebuild-changed     Rebuild only changed stages"
    echo "  help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  bash scripts/manage_dvc_cache.sh status"
    echo "  bash scripts/manage_dvc_cache.sh clear"
    echo "  bash scripts/manage_dvc_cache.sh rebuild"
}

# Main
case "${1:-status}" in
    status)
        show_status
        ;;
    clear)
        clear_cache
        ;;
    list)
        list_cached
        ;;
    rebuild)
        rebuild
        ;;
    rebuild-changed)
        rebuild_changed
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac
