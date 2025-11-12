#!/bin/bash
# Evaluate model against all target types

MODEL_PATH="$1"

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: ./evaluate_all_targets.sh <model_path>"
    echo "Example: ./evaluate_all_targets.sh ./models/experiment/best_model.zip"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

echo "=================================="
echo "Evaluating All Target Types"
echo "=================================="
echo "Model: $MODEL_PATH"
echo ""

# Her hedef tipi için evaluate
for maneuver in straight circular zigzag evasive; do
    echo ""
    echo "Testing against $maneuver target..."
    echo "=================================="

    python evaluate.py \
        --model_path "$MODEL_PATH" \
        --target_maneuver "$maneuver" \
        --n_episodes 10 \
        --output_dir "./evaluation_results/${maneuver}"

    echo "✓ $maneuver completed"
done

echo ""
echo "=================================="
echo "All Evaluations Complete!"
echo "=================================="
echo ""
echo "Results saved in ./evaluation_results/"
echo ""
echo "Summary plots:"
ls -1 ./evaluation_results/*/evaluation_summary.png 2>/dev/null
echo ""
echo "Trajectory plots:"
ls -1 ./evaluation_results/*/trajectory_episode_*.png 2>/dev/null | head -4
