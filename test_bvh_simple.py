#!/usr/bin/env python3
"""
Simple BVH Testing Script
Tests generated BVH files by comparing basic characteristics with reference files.
"""

import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple


def parse_bvh_basic(filepath: Path) -> Dict:
    """Parse BVH file and extract basic information"""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find frame count and frame time
    frames = 0
    frame_time = 0.0
    motion_start = 0
    joint_count = 0

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('JOINT') or line.startswith('ROOT'):
            joint_count += 1
        elif line == 'MOTION':
            motion_start = i
        elif line.startswith('Frames:'):
            frames = int(line.split()[1])
        elif line.startswith('Frame Time:'):
            frame_time = float(line.split()[2])

    # Count motion data lines
    motion_lines = 0
    for i in range(motion_start + 3, len(lines)):
        if lines[i].strip() and not lines[i].startswith('#'):
            motion_lines += 1

    # Get first and last motion frames for basic analysis
    first_frame = None
    last_frame = None

    for i in range(motion_start + 3, len(lines)):
        line = lines[i].strip()
        if line and not line.startswith('#'):
            values = [float(v) for v in line.split()]
            if first_frame is None:
                first_frame = values
            last_frame = values

    return {
        'frames': frames,
        'frame_time': frame_time,
        'joint_count': joint_count,
        'motion_lines': motion_lines,
        'first_frame': first_frame,
        'last_frame': last_frame,
        'duration': frames * frame_time if frame_time > 0 else 0
    }


def analyze_motion_basic(bvh_data: Dict) -> Dict:
    """Analyze basic motion characteristics"""
    results = {}

    # Basic stats
    results['frame_count'] = bvh_data['frames']
    results['duration_seconds'] = bvh_data['duration']
    results['joint_count'] = bvh_data['joint_count']

    # Calculate displacement (using first 3 values as root position)
    if bvh_data['first_frame'] and bvh_data['last_frame']:
        if len(bvh_data['first_frame']) >= 3 and len(bvh_data['last_frame']) >= 3:
            displacement = [
                bvh_data['last_frame'][0] - bvh_data['first_frame'][0],  # X
                bvh_data['last_frame'][1] - bvh_data['first_frame'][1],  # Y
                bvh_data['last_frame'][2] - bvh_data['first_frame'][2],  # Z
            ]
            results['total_displacement'] = np.sqrt(displacement[0]**2 + displacement[2]**2)
            results['vertical_change'] = displacement[1]

            # Calculate average speed
            if bvh_data['duration'] > 0:
                results['average_speed'] = results['total_displacement'] / bvh_data['duration']
            else:
                results['average_speed'] = 0.0

    # Check data consistency
    results['expected_frames'] = bvh_data['frames']
    results['actual_motion_lines'] = bvh_data['motion_lines']
    results['data_complete'] = (bvh_data['frames'] == bvh_data['motion_lines'])

    return results


def compare_bvh_files(generated_path: Path, reference_path: Path) -> Dict:
    """Compare two BVH files and return similarity metrics"""

    print(f"\nComparing BVH files:")
    print(f"  Generated: {generated_path.name}")
    print(f"  Reference: {reference_path.name}")

    # Parse both files
    gen_data = parse_bvh_basic(generated_path)
    ref_data = parse_bvh_basic(reference_path)

    # Analyze both
    gen_analysis = analyze_motion_basic(gen_data)
    ref_analysis = analyze_motion_basic(ref_data)

    # Compare characteristics
    comparison = {}

    # Duration comparison
    if ref_analysis['duration_seconds'] > 0:
        duration_ratio = gen_analysis['duration_seconds'] / ref_analysis['duration_seconds']
        comparison['duration_match'] = max(0, 100 * (1 - abs(1 - duration_ratio)))
    else:
        comparison['duration_match'] = 0

    # Frame count comparison
    frame_diff = abs(gen_analysis['frame_count'] - ref_analysis['frame_count'])
    comparison['frame_count_diff'] = frame_diff

    # Joint count comparison
    comparison['joint_count_match'] = (gen_analysis['joint_count'] == ref_analysis['joint_count'])

    # Speed comparison (if available)
    if 'average_speed' in gen_analysis and 'average_speed' in ref_analysis:
        if ref_analysis['average_speed'] > 0:
            speed_ratio = gen_analysis['average_speed'] / ref_analysis['average_speed']
            comparison['speed_similarity'] = max(0, 100 * (1 - abs(1 - speed_ratio) / 2))
        else:
            comparison['speed_similarity'] = 50

    # Data completeness
    comparison['generated_complete'] = gen_analysis['data_complete']
    comparison['reference_complete'] = ref_analysis['data_complete']

    # Overall score
    scores = []
    if 'duration_match' in comparison:
        scores.append(comparison['duration_match'])
    if 'speed_similarity' in comparison:
        scores.append(comparison['speed_similarity'])
    if comparison['joint_count_match']:
        scores.append(100)
    else:
        scores.append(0)

    comparison['overall_score'] = np.mean(scores) if scores else 0

    # Add analysis details
    comparison['generated_analysis'] = gen_analysis
    comparison['reference_analysis'] = ref_analysis

    return comparison


def main():
    print("="*60)
    print("BVH Reference Testing System")
    print("="*60)

    # Define reference files
    reference_files = [
        Path('bvh_examples/walk-through-spce.bvh'),
        Path('bvh_examples/walking-standing-still.bvh')
    ]

    # Check which reference files exist
    available_refs = []
    for ref_file in reference_files:
        if ref_file.exists():
            available_refs.append(ref_file)
            print(f"✓ Found reference: {ref_file.name}")

            # Analyze reference
            ref_data = parse_bvh_basic(ref_file)
            ref_analysis = analyze_motion_basic(ref_data)

            print(f"  - Frames: {ref_analysis['frame_count']}")
            print(f"  - Duration: {ref_analysis['duration_seconds']:.2f} seconds")
            print(f"  - Joints: {ref_analysis['joint_count']}")
            if 'total_displacement' in ref_analysis:
                print(f"  - Displacement: {ref_analysis['total_displacement']:.2f} units")
            if 'average_speed' in ref_analysis:
                print(f"  - Speed: {ref_analysis['average_speed']:.2f} units/sec")
        else:
            print(f"✗ Reference not found: {ref_file}")

    if not available_refs:
        print("\nNo reference files found!")
        return

    # Find generated BVH files to test
    test_candidates = [
        Path('bvh/walking_spine_fixed.bvh'),
        Path('bvh/walking_00001.bvh'),
        Path('bvh/thewave_spine_fixed.bvh'),
        Path('bvh/boxer_spine_fixed.bvh'),
        Path('bvh/shrug.bvh'),
        Path('bvh/thewave.bvh')
    ]

    print("\n" + "="*60)
    print("Testing Generated BVH Files")
    print("="*60)

    results = []

    for test_file in test_candidates:
        if test_file.exists():
            print(f"\nTesting: {test_file.name}")
            print("-"*40)

            # Test against first available reference
            comparison = compare_bvh_files(test_file, available_refs[0])

            # Print results
            print(f"Overall Score: {comparison['overall_score']:.1f}%")

            if comparison['overall_score'] >= 80:
                grade = 'A'
                assessment = 'Excellent'
            elif comparison['overall_score'] >= 70:
                grade = 'B'
                assessment = 'Good'
            elif comparison['overall_score'] >= 60:
                grade = 'C'
                assessment = 'Fair'
            elif comparison['overall_score'] >= 50:
                grade = 'D'
                assessment = 'Poor'
            else:
                grade = 'F'
                assessment = 'Failed'

            print(f"Grade: {grade} ({assessment})")

            print("\nDetails:")
            print(f"  Duration match: {comparison.get('duration_match', 0):.1f}%")
            print(f"  Frame difference: {comparison['frame_count_diff']} frames")
            print(f"  Joint count match: {'Yes' if comparison['joint_count_match'] else 'No'}")

            if 'speed_similarity' in comparison:
                print(f"  Speed similarity: {comparison['speed_similarity']:.1f}%")

            print(f"  Data complete: {'Yes' if comparison['generated_complete'] else 'No'}")

            # Store results
            results.append({
                'file': str(test_file),
                'score': comparison['overall_score'],
                'grade': grade,
                'comparison': comparison
            })

    # Save results
    output_dir = Path('comparison_results')
    output_dir.mkdir(exist_ok=True)

    if results:
        results_file = output_dir / 'test_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n{'='*60}")
        print(f"Results saved to: {results_file}")

        # Summary
        print("\nSummary:")
        print("-"*40)
        for result in results:
            print(f"{Path(result['file']).name:30} Grade: {result['grade']} ({result['score']:.1f}%)")
    else:
        print("\nNo test files found!")


if __name__ == "__main__":
    main()