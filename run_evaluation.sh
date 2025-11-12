#!/bin/bash
# Automatic model evaluation script

echo "=================================="
echo "Automatic Model Evaluation"
echo "=================================="
echo ""

# Model path parametresi varsa kullan, yoksa otomatik bul
if [ -n "$1" ]; then
    MODEL_PATH="$1"
    echo "Using provided model: $MODEL_PATH"
else
    echo "Searching for models..."

    # En son best_model'i bul
    BEST_MODEL=$(find ./models -name 'best_model.zip' -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

    # Yoksa final_model'i bul
    if [ -z "$BEST_MODEL" ]; then
        BEST_MODEL=$(find ./models -name 'final_model.zip' -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    fi

    # Hala yoksa herhangi bir .zip bul
    if [ -z "$BEST_MODEL" ]; then
        BEST_MODEL=$(find ./models -name '*.zip' -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    fi

    if [ -z "$BEST_MODEL" ]; then
        echo "❌ No model found in ./models/"
        echo ""
        echo "Please provide model path manually:"
        echo "  ./run_evaluation.sh /path/to/your/model.zip"
        exit 1
    fi

    MODEL_PATH="$BEST_MODEL"
    echo "✓ Found model: $MODEL_PATH"
fi

echo ""

# Dosya var mı kontrol et
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model file not found: $MODEL_PATH"
    exit 1
fi

echo "=================================="
echo "Starting Evaluation..."
echo "=================================="
echo ""

# Hedef tipi parametresi (varsayılan: straight)
TARGET=${2:-straight}
N_EPISODES=${3:-10}

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Target: $TARGET"
echo "  Episodes: $N_EPISODES"
echo ""

# Evaluation çalıştır
python evaluate.py \
    --model_path "$MODEL_PATH" \
    --target_maneuver "$TARGET" \
    --n_episodes "$N_EPISODES" \
    --output_dir "./evaluation_results/${TARGET}"

echo ""
echo "=================================="
echo "Evaluation Complete!"
echo "=================================="
echo "Results saved to: ./evaluation_results/${TARGET}"
echo ""
echo "To evaluate all target types, run:"
echo "  ./evaluate_all_targets.sh \"$MODEL_PATH\""
