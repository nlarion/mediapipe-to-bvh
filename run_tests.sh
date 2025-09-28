#!/bin/bash

# BVH Testing Suite Runner
# This script helps test video-to-BVH conversion accuracy

echo "======================================"
echo "Video to BVH Testing Suite"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default directories
VIDEO_DIR="videos"
BVH_DIR="bvh"
TEST_RESULTS_DIR="test_results"

# Create test results directory if it doesn't exist
mkdir -p "$TEST_RESULTS_DIR"

# Function to print colored messages
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to test a single video
test_single_video() {
    local video_file=$1
    local video_name=$(basename "$video_file" .mp4)
    
    print_status "Testing: $video_file"
    
    # Run accuracy test
    python3 test_bvh_accuracy.py --video "$video_file" --output-dir "$TEST_RESULTS_DIR"
    
    # Check if BVH exists for comparison
    local bvh_file="${BVH_DIR}/${video_name}.bvh"
    if [ -f "$bvh_file" ]; then
        print_status "Comparing with existing BVH: $bvh_file"
        python3 compare_bvh_output.py \
            --video "$video_file" \
            --bvh "$bvh_file" \
            --output "${TEST_RESULTS_DIR}/${video_name}_comparison.png"
    else
        print_warning "No BVH file found for comparison: $bvh_file"
        print_status "Generating BVH file..."
        python3 mediapipe_to_bvh_complete.py \
            --video "$video_file" \
            --output "$bvh_file" \
            --sample-rate 2
        
        if [ -f "$bvh_file" ]; then
            print_status "BVH generated, running comparison..."
            python3 compare_bvh_output.py \
                --video "$video_file" \
                --bvh "$bvh_file" \
                --output "${TEST_RESULTS_DIR}/${video_name}_comparison.png"
        fi
    fi
}

# Function to run batch tests
run_batch_tests() {
    print_status "Running batch tests on all videos in $VIDEO_DIR"
    python3 test_bvh_accuracy.py --video-dir "$VIDEO_DIR" --output-dir "$TEST_RESULTS_DIR"
}

# Function to test specific video types
test_by_category() {
    local category=$1
    
    print_status "Testing $category videos..."
    
    case $category in
        "walking")
            pattern="walking*.mp4"
            ;;
        "boxing")
            pattern="boxer*.mp4"
            ;;
        "soccer")
            pattern="soccer*.mp4"
            ;;
        "soldier")
            pattern="soldier*.mp4"
            ;;
        *)
            pattern="*.mp4"
            ;;
    esac
    
    python3 test_bvh_accuracy.py \
        --video-dir "$VIDEO_DIR" \
        --pattern "$pattern" \
        --output-dir "$TEST_RESULTS_DIR"
}

# Main menu
show_menu() {
    echo ""
    echo "Select testing option:"
    echo "1) Test single video"
    echo "2) Test all videos (batch)"
    echo "3) Test walking videos"
    echo "4) Test boxing videos"
    echo "5) Test soccer videos"
    echo "6) Quick test (first 3 videos)"
    echo "7) Generate comparison for existing BVH"
    echo "8) Exit"
    echo ""
}

# Quick test function
quick_test() {
    print_status "Running quick test on first 3 videos..."
    
    count=0
    for video in "$VIDEO_DIR"/*.mp4; do
        if [ -f "$video" ]; then
            test_single_video "$video"
            count=$((count + 1))
            if [ $count -ge 3 ]; then
                break
            fi
        fi
    done
}

# Main script logic
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: $0 [option]"
    echo "Options:"
    echo "  --single <video>  Test a single video"
    echo "  --batch          Run batch tests"
    echo "  --quick          Quick test (first 3 videos)"
    echo "  --category <cat> Test videos by category"
    echo ""
    exit 0
fi

# Handle command line arguments
if [ "$1" == "--single" ] && [ -n "$2" ]; then
    test_single_video "$2"
elif [ "$1" == "--batch" ]; then
    run_batch_tests
elif [ "$1" == "--quick" ]; then
    quick_test
elif [ "$1" == "--category" ] && [ -n "$2" ]; then
    test_by_category "$2"
else
    # Interactive menu
    while true; do
        show_menu
        read -p "Enter choice [1-8]: " choice
        
        case $choice in
            1)
                read -p "Enter video path: " video_path
                if [ -f "$video_path" ]; then
                    test_single_video "$video_path"
                else
                    print_error "Video file not found: $video_path"
                fi
                ;;
            2)
                run_batch_tests
                ;;
            3)
                test_by_category "walking"
                ;;
            4)
                test_by_category "boxing"
                ;;
            5)
                test_by_category "soccer"
                ;;
            6)
                quick_test
                ;;
            7)
                read -p "Enter video path: " video_path
                read -p "Enter BVH path: " bvh_path
                if [ -f "$video_path" ] && [ -f "$bvh_path" ]; then
                    video_name=$(basename "$video_path" .mp4)
                    python3 compare_bvh_output.py \
                        --video "$video_path" \
                        --bvh "$bvh_path" \
                        --output "${TEST_RESULTS_DIR}/${video_name}_comparison.png"
                else
                    print_error "File not found"
                fi
                ;;
            8)
                print_status "Exiting..."
                exit 0
                ;;
            *)
                print_error "Invalid option"
                ;;
        esac
    done
fi

print_status "Testing complete! Results saved in $TEST_RESULTS_DIR/"