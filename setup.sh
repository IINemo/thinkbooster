#!/bin/bash
# Setup script for ThinkBooster
# Installs package dependencies, lm-polygraph dev branch, and llm-uncertainty-head (luh)

set -e  # Exit on error

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LM_POLYGRAPH_DIR="$SCRIPT_DIR/lm-polygraph"
KERNELACT_DIR="$SCRIPT_DIR/thinkbooster/datasets/KernelAct"
LUH_DIR="$SCRIPT_DIR/llm-uncertainty-head"

# Parse arguments
UPDATE_ONLY=false
VERBOSE=false
for arg in "$@"; do
    case "$arg" in
        --update|-u) UPDATE_ONLY=true ;;
        --verbose|-v) VERBOSE=true ;;
    esac
done

# Redirect pip output based on verbosity
pip_install() {
    if [ "$VERBOSE" = true ]; then
        pip install "$@"
    else
        pip install "$@" > /dev/null 2>&1
    fi
}

install_lm_polygraph() {
    echo -e "${YELLOW}Setting up lm-polygraph...${NC}"

    if [ -d "$LM_POLYGRAPH_DIR" ]; then
        echo -e "  Pulling latest changes..."
        cd "$LM_POLYGRAPH_DIR"
        git pull origin main 2>&1 | grep -E "(Already|Updating)" || true
        cd "$SCRIPT_DIR"
    else
        echo -e "  Cloning lm-polygraph..."
        git clone https://github.com/IINemo/lm-polygraph.git
    fi

    echo -e "  Installing lm-polygraph..."
    pip_install -e "$LM_POLYGRAPH_DIR"
    echo -e "${GREEN}✓ lm-polygraph installed${NC}"
}

install_luh() {
    echo -e "${YELLOW}Setting up llm-uncertainty-head (luh)...${NC}"

    if [ -d "$LUH_DIR" ]; then
        echo -e "  Pulling latest changes..."
        cd "$LUH_DIR"
        git pull origin main 2>&1 | grep -E "(Already|Updating)" || true
        cd "$SCRIPT_DIR"
    else
        echo -e "  Cloning llm-uncertainty-head..."
        git clone https://github.com/IINemo/llm-uncertainty-head.git "$LUH_DIR"
    fi

    # vllm-speculators is required for hidden states extraction
    echo -e "  Installing vllm-speculators (hidden states support)..."
    pip_install "git+https://github.com/vllm-project/speculators.git"
    echo -e "${GREEN}✓ vllm-speculators installed${NC}"

    echo -e "  Installing luh..."
    pip_install -e "$LUH_DIR"
    echo -e "${GREEN}✓ luh installed${NC}"
}

install_kernelact() {
    echo -e "${YELLOW}Setting up KernelAct (feat/tts-service-integration branch)...${NC}"

    if [ -d "$KERNELACT_DIR" ]; then
        echo -e "  Pulling latest changes..."
        cd "$KERNELACT_DIR"
        git pull origin feat/tts-service-integration 2>&1 | grep -E "(Already|Updating)" || true
        cd "$SCRIPT_DIR"
    else
        echo -e "  Cloning KernelAct to thinkbooster/datasets..."
        if ! git clone -b feat/tts-service-integration https://github.com/ai-nikolai/KernelAct.git "$KERNELACT_DIR" 2>/dev/null; then
            echo -e "${RED}✗ Failed to clone KernelAct (skipping)${NC}"
            echo -e "  ${YELLOW}Note: KernelAct is optional, required only for KernelBench dataset${NC}"
            return 0
        fi
    fi
    echo -e "${GREEN}✓ KernelAct cloned${NC}"
}

if [ "$UPDATE_ONLY" = true ]; then
    install_lm_polygraph
    install_kernelact
    install_luh
    exit 0
fi

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  ThinkBooster Setup${NC}"
echo -e "${BLUE}======================================${NC}\n"

# Install package dependencies via pip
echo -e "${YELLOW}Installing package dependencies...${NC}"
pip_install -e .
pip_install -e ".[vllm]"                          # vLLM for fast local inference
pip_install latex2sympy2 --no-deps                # math evaluation (separate due to antlr4 conflict with hydra)
echo -e "${GREEN}✓ Package installed${NC}\n"

# Install lm-polygraph dev branch
install_lm_polygraph

# Install KernelAct for KernelBench prompt generation (cloned to thinkbooster/datasets)
install_kernelact

# Install llm-uncertainty-head (luh) for UHead scorer
install_luh

echo -e "\n${GREEN}✅ Setup complete!${NC}"
echo -e "\nNext: Copy .env.example to .env and add your API keys"
echo -e "Update dependencies: ${BLUE}./setup.sh --update${NC}"
