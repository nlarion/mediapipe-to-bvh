#!/usr/bin/env python3
"""
BVH Testing with Reference Motion Comparison

This script tests generated BVH files by comparing them against known good reference motions.
It evaluates how well the generated motion matches the characteristics of professional BVH files.
"""

import numpy as np
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime
from bvh_reference_analyzer import BVHReferenceAnalyzer, ReferenceMotionProfile
from dataclasses import dataclass


@dataclass
class ComparisonMetrics:
    """Metrics comparing generated BVH to reference motion"""
    # Gait similarity
    stride_length_similarity: float  # 0-100%
    stride_frequency_similarity: float
    gait_pattern_score: float

    # Motion quality
    smoothness_comparison: float  # How similar the jerk metrics are
    oscillation_similarity: float  # Vertical motion patterns

    # Joint angle similarity
    joint_angle_correlation: float  # Overall correlation
    joint_range_similarity: Dict[str, float]  # Per-joint similarity

    # Timing similarity
    phase_duration_similarity: float
    rhythm_consistency: float

    # Energy profile similarity
    energy_correlation: float
    dynamics_similarity: float

    # Overall scores
    overall_similarity: float  # 0-100%
    quality_grade: str  # A, B, C, D, F
    detailed_feedback: List[str]


class BVHReferenceComparator:
    """Compares generated BVH against reference motions"""

    def __init__(self):
        self.analyzer = BVHReferenceAnalyzer()

    def compare_to_reference(
        self,
        generated_bvh_path: Path,
        reference_profile: ReferenceMotionProfile
    ) -> ComparisonMetrics:
        """Compare generated BVH to reference motion profile"""

        # Analyze the generated BVH
        generated_profile = self.analyzer.analyze_reference_motion(generated_bvh_path)

        # Compare stride characteristics
        stride_length_similarity = self._calculate_similarity(
            generated_profile.stride_length,
            reference_profile.stride_length,
            tolerance=0.3
        )

        stride_frequency_similarity = self._calculate_similarity(
            generated_profile.stride_frequency,
            reference_profile.stride_frequency,
            tolerance=0.2
        )

        # Compare motion smoothness
        smoothness_comparison = self._calculate_similarity(
            generated_profile.jerk_metric,
            reference_profile.jerk_metric,
            tolerance=0.5,
            lower_is_better=True
        )

        # Compare vertical oscillation
        oscillation_similarity = self._calculate_similarity(
            generated_profile.hip_vertical_displacement,
            reference_profile.hip_vertical_displacement,
            tolerance=0.2
        )

        # Compare joint angles
        joint_range_similarity = {}
        common_joints = set(generated_profile.joint_angle_ranges.keys()) & \
                       set(reference_profile.joint_angle_ranges.keys())

        for joint in common_joints:
            gen_range = generated_profile.joint_angle_ranges[joint]
            ref_range = reference_profile.joint_angle_ranges[joint]

            range_diff_min = abs(gen_range[0] - ref_range[0])
            range_diff_max = abs(gen_range[1] - ref_range[1])

            similarity = max(0, 100 - (range_diff_min + range_diff_max) / 2)
            joint_range_similarity[joint] = similarity

        joint_angle_correlation = np.mean(list(joint_range_similarity.values())) if joint_range_similarity else 0

        # Compare phase durations
        phase_duration_similarity = self._calculate_similarity(
            generated_profile.stance_phase_duration,
            reference_profile.stance_phase_duration,
            tolerance=0.1
        )

        # Calculate rhythm consistency
        rhythm_consistency = self._evaluate_rhythm(
            generated_profile,
            reference_profile
        )

        # Compare energy profiles
        energy_correlation = self._calculate_energy_correlation(
            generated_profile,
            reference_profile
        )

        # Calculate dynamics similarity
        dynamics_similarity = self._calculate_dynamics_similarity(
            generated_profile,
            reference_profile
        )

        # Calculate gait pattern score
        gait_pattern_score = np.mean([
            stride_length_similarity,
            stride_frequency_similarity,
            phase_duration_similarity
        ])

        # Calculate overall similarity
        overall_similarity = np.mean([
            stride_length_similarity * 0.15,
            stride_frequency_similarity * 0.15,
            smoothness_comparison * 0.20,
            oscillation_similarity * 0.10,
            joint_angle_correlation * 0.20,
            phase_duration_similarity * 0.10,
            energy_correlation * 0.10
        ])

        # Determine quality grade
        quality_grade = self._determine_grade(overall_similarity)

        # Generate detailed feedback
        detailed_feedback = self._generate_feedback(
            generated_profile,
            reference_profile,
            overall_similarity,
            {
                'stride_length': stride_length_similarity,
                'stride_frequency': stride_frequency_similarity,
                'smoothness': smoothness_comparison,
                'oscillation': oscillation_similarity,
                'joint_angles': joint_angle_correlation
            }
        )

        return ComparisonMetrics(
            stride_length_similarity=stride_length_similarity,
            stride_frequency_similarity=stride_frequency_similarity,
            gait_pattern_score=gait_pattern_score,
            smoothness_comparison=smoothness_comparison,
            oscillation_similarity=oscillation_similarity,
            joint_angle_correlation=joint_angle_correlation,
            joint_range_similarity=joint_range_similarity,
            phase_duration_similarity=phase_duration_similarity,
            rhythm_consistency=rhythm_consistency,
            energy_correlation=energy_correlation,
            dynamics_similarity=dynamics_similarity,
            overall_similarity=overall_similarity,
            quality_grade=quality_grade,
            detailed_feedback=detailed_feedback
        )

    def _calculate_similarity(
        self,
        value1: float,
        value2: float,
        tolerance: float = 0.2,
        lower_is_better: bool = False
    ) -> float:
        """Calculate similarity percentage between two values"""
        if value2 == 0:
            return 100.0 if value1 == 0 else 0.0

        ratio = value1 / value2
        if lower_is_better:
            # For metrics where lower is better (like jerk)
            if value1 <= value2:
                return 100.0
            else:
                diff = abs(1 - ratio)
                return max(0, 100 * (1 - diff / tolerance))
        else:
            diff = abs(1 - ratio)
            return max(0, 100 * (1 - diff / tolerance))

    def _evaluate_rhythm(
        self,
        generated: ReferenceMotionProfile,
        reference: ReferenceMotionProfile
    ) -> float:
        """Evaluate rhythm consistency between profiles"""
        # Simple rhythm evaluation based on timing patterns
        if reference.stride_frequency > 0:
            freq_ratio = generated.stride_frequency / reference.stride_frequency
            rhythm_score = 100 * np.exp(-abs(1 - freq_ratio))
        else:
            rhythm_score = 50.0

        return rhythm_score

    def _calculate_energy_correlation(
        self,
        generated: ReferenceMotionProfile,
        reference: ReferenceMotionProfile
    ) -> float:
        """Calculate correlation between energy profiles"""
        if len(generated.kinetic_energy_pattern) == 0 or len(reference.kinetic_energy_pattern) == 0:
            return 50.0

        # Normalize patterns to same length
        min_len = min(len(generated.kinetic_energy_pattern), len(reference.kinetic_energy_pattern))

        if min_len > 1:
            gen_norm = generated.kinetic_energy_pattern[:min_len]
            ref_norm = reference.kinetic_energy_pattern[:min_len]

            # Normalize magnitudes
            if np.std(ref_norm) > 0 and np.std(gen_norm) > 0:
                correlation = np.corrcoef(gen_norm, ref_norm)[0, 1]
                return max(0, correlation * 100)

        return 50.0

    def _calculate_dynamics_similarity(
        self,
        generated: ReferenceMotionProfile,
        reference: ReferenceMotionProfile
    ) -> float:
        """Calculate similarity in motion dynamics"""
        scores = []

        # Compare acceleration profiles
        if len(generated.acceleration_profile) > 0 and len(reference.acceleration_profile) > 0:
            min_len = min(len(generated.acceleration_profile), len(reference.acceleration_profile))
            gen_acc = generated.acceleration_profile[:min_len]
            ref_acc = reference.acceleration_profile[:min_len]

            if np.std(ref_acc) > 0:
                acc_diff = np.mean(np.abs(gen_acc - ref_acc))
                acc_score = max(0, 100 - acc_diff * 10)
                scores.append(acc_score)

        # Compare jerk metrics
        jerk_score = self._calculate_similarity(
            generated.jerk_metric,
            reference.jerk_metric,
            tolerance=0.5,
            lower_is_better=True
        )
        scores.append(jerk_score)

        return np.mean(scores) if scores else 50.0

    def _determine_grade(self, overall_similarity: float) -> str:
        """Determine letter grade based on similarity score"""
        if overall_similarity >= 90:
            return 'A'
        elif overall_similarity >= 80:
            return 'B'
        elif overall_similarity >= 70:
            return 'C'
        elif overall_similarity >= 60:
            return 'D'
        else:
            return 'F'

    def _generate_feedback(
        self,
        generated: ReferenceMotionProfile,
        reference: ReferenceMotionProfile,
        overall_score: float,
        component_scores: Dict[str, float]
    ) -> List[str]:
        """Generate detailed feedback about the comparison"""
        feedback = []

        # Overall assessment
        if overall_score >= 80:
            feedback.append("Excellent match to reference motion!")
        elif overall_score >= 70:
            feedback.append("Good match with minor differences from reference.")
        elif overall_score >= 60:
            feedback.append("Moderate match - some aspects need improvement.")
        else:
            feedback.append("Significant differences from reference motion.")

        # Specific feedback
        if component_scores['stride_length'] < 70:
            diff = generated.stride_length - reference.stride_length
            if diff > 0:
                feedback.append(f"Stride length is {abs(diff):.2f} units longer than reference.")
            else:
                feedback.append(f"Stride length is {abs(diff):.2f} units shorter than reference.")

        if component_scores['stride_frequency'] < 70:
            diff = generated.stride_frequency - reference.stride_frequency
            if diff > 0:
                feedback.append(f"Walking pace is {abs(diff):.2f} Hz faster than reference.")
            else:
                feedback.append(f"Walking pace is {abs(diff):.2f} Hz slower than reference.")

        if component_scores['smoothness'] < 70:
            if generated.jerk_metric > reference.jerk_metric * 1.5:
                feedback.append("Motion appears less smooth than reference (higher jerk).")
            else:
                feedback.append("Motion smoothness differs from reference pattern.")

        if component_scores['oscillation'] < 70:
            diff = generated.hip_vertical_displacement - reference.hip_vertical_displacement
            if abs(diff) > reference.hip_vertical_displacement * 0.3:
                feedback.append(f"Vertical motion differs by {abs(diff):.2f} units from reference.")

        if component_scores['joint_angles'] < 70:
            feedback.append("Joint angle ranges differ significantly from reference motion.")

        # Positive feedback for good matches
        if component_scores['smoothness'] >= 85:
            feedback.append("Motion smoothness matches reference well.")

        if component_scores['stride_frequency'] >= 85:
            feedback.append("Walking rhythm closely matches reference.")

        return feedback

    def visualize_comparison(
        self,
        generated_bvh_path: Path,
        reference_profile: ReferenceMotionProfile,
        metrics: ComparisonMetrics,
        output_path: Optional[Path] = None
    ):
        """Create visualization of the comparison"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'BVH Comparison - Grade: {metrics.quality_grade} ({metrics.overall_similarity:.1f}%)', fontsize=14)

        # Component scores bar chart
        ax = axes[0, 0]
        components = ['Stride\nLength', 'Stride\nFreq', 'Smoothness', 'Oscillation', 'Joint\nAngles']
        scores = [
            metrics.stride_length_similarity,
            metrics.stride_frequency_similarity,
            metrics.smoothness_comparison,
            metrics.oscillation_similarity,
            metrics.joint_angle_correlation
        ]

        bars = ax.bar(components, scores)
        for bar, score in zip(bars, scores):
            color = 'green' if score >= 80 else 'yellow' if score >= 60 else 'red'
            bar.set_color(color)

        ax.set_ylim(0, 100)
        ax.set_ylabel('Similarity %')
        ax.set_title('Component Scores')
        ax.axhline(y=80, color='g', linestyle='--', alpha=0.3)
        ax.axhline(y=60, color='y', linestyle='--', alpha=0.3)

        # Joint angle similarity heatmap
        ax = axes[0, 1]
        if metrics.joint_range_similarity:
            joints = list(metrics.joint_range_similarity.keys())[:10]  # Limit to 10 joints
            similarities = [metrics.joint_range_similarity[j] for j in joints]

            im = ax.imshow([similarities], aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
            ax.set_xticks(range(len(joints)))
            ax.set_xticklabels(joints, rotation=45, ha='right')
            ax.set_yticks([])
            ax.set_title('Joint Angle Similarity')
            plt.colorbar(im, ax=ax)

        # Overall metrics radar chart
        ax = axes[0, 2]
        categories = ['Gait', 'Smooth', 'Rhythm', 'Energy', 'Dynamics']
        values = [
            metrics.gait_pattern_score,
            metrics.smoothness_comparison,
            metrics.rhythm_consistency,
            metrics.energy_correlation,
            metrics.dynamics_similarity
        ]

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        values = values + [values[0]]  # Complete the circle
        angles = np.concatenate([angles, [angles[0]]])

        ax = plt.subplot(2, 3, 3, projection='polar')
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_title('Motion Quality Profile')

        # Feedback text
        ax = axes[1, 0]
        ax.axis('off')
        feedback_text = '\n'.join(metrics.detailed_feedback[:5])  # Show first 5 feedback items
        ax.text(0.1, 0.9, 'Feedback:', fontweight='bold', transform=ax.transAxes)
        ax.text(0.1, 0.8, feedback_text, transform=ax.transAxes, verticalalignment='top', fontsize=9)

        # Grade visualization
        ax = axes[1, 1]
        ax.axis('off')
        grade_color = {
            'A': 'green',
            'B': 'yellowgreen',
            'C': 'yellow',
            'D': 'orange',
            'F': 'red'
        }[metrics.quality_grade]

        ax.text(0.5, 0.5, metrics.quality_grade, fontsize=72, fontweight='bold',
                color=grade_color, ha='center', va='center', transform=ax.transAxes)
        ax.text(0.5, 0.2, f'{metrics.overall_similarity:.1f}%', fontsize=24,
                ha='center', va='center', transform=ax.transAxes)

        # Summary stats
        ax = axes[1, 2]
        ax.axis('off')
        summary = f"""Summary Statistics:

Gait Pattern: {metrics.gait_pattern_score:.1f}%
Motion Quality: {metrics.smoothness_comparison:.1f}%
Joint Accuracy: {metrics.joint_angle_correlation:.1f}%
Energy Match: {metrics.energy_correlation:.1f}%
Overall: {metrics.overall_similarity:.1f}%"""

        ax.text(0.1, 0.5, summary, transform=ax.transAxes, fontsize=10, verticalalignment='center')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=100, bbox_inches='tight')

        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Test BVH files against reference motions')
    parser.add_argument('generated_bvh', type=str, help='Path to generated BVH file')
    parser.add_argument('--reference', type=str, help='Path to reference BVH or profile JSON')
    parser.add_argument('--output-dir', type=str, default='comparison_results',
                       help='Directory for output files')
    parser.add_argument('--visualize', action='store_true', help='Create visualization')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load or create reference profile
    analyzer = BVHReferenceAnalyzer()
    reference_path = Path(args.reference) if args.reference else Path('bvh_examples/walk-through-spce.bvh')

    if reference_path.suffix == '.json':
        print(f"Loading reference profile from {reference_path}")
        reference_profile = analyzer.load_profile(reference_path)
    elif reference_path.suffix == '.bvh':
        print(f"Analyzing reference BVH: {reference_path}")
        reference_profile = analyzer.analyze_reference_motion(reference_path)
    else:
        print("Invalid reference file format. Use .bvh or .json")
        return

    # Compare generated BVH to reference
    comparator = BVHReferenceComparator()
    generated_path = Path(args.generated_bvh)

    print(f"\nComparing {generated_path.name} to reference...")
    metrics = comparator.compare_to_reference(generated_path, reference_profile)

    # Print results
    print(f"\n{'='*50}")
    print(f"COMPARISON RESULTS - Grade: {metrics.quality_grade}")
    print(f"{'='*50}")
    print(f"Overall Similarity: {metrics.overall_similarity:.1f}%")
    print(f"\nComponent Scores:")
    print(f"  Stride Length:     {metrics.stride_length_similarity:.1f}%")
    print(f"  Stride Frequency:  {metrics.stride_frequency_similarity:.1f}%")
    print(f"  Motion Smoothness: {metrics.smoothness_comparison:.1f}%")
    print(f"  Vertical Motion:   {metrics.oscillation_similarity:.1f}%")
    print(f"  Joint Angles:      {metrics.joint_angle_correlation:.1f}%")
    print(f"\nQuality Metrics:")
    print(f"  Gait Pattern:      {metrics.gait_pattern_score:.1f}%")
    print(f"  Rhythm:            {metrics.rhythm_consistency:.1f}%")
    print(f"  Energy Profile:    {metrics.energy_correlation:.1f}%")
    print(f"  Dynamics:          {metrics.dynamics_similarity:.1f}%")

    print(f"\nFeedback:")
    for feedback in metrics.detailed_feedback:
        print(f"  • {feedback}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"comparison_{generated_path.stem}_{timestamp}.json"

    results_dict = {
        'generated_bvh': str(generated_path),
        'reference': str(reference_path),
        'timestamp': timestamp,
        'overall_similarity': metrics.overall_similarity,
        'quality_grade': metrics.quality_grade,
        'component_scores': {
            'stride_length_similarity': metrics.stride_length_similarity,
            'stride_frequency_similarity': metrics.stride_frequency_similarity,
            'gait_pattern_score': metrics.gait_pattern_score,
            'smoothness_comparison': metrics.smoothness_comparison,
            'oscillation_similarity': metrics.oscillation_similarity,
            'joint_angle_correlation': metrics.joint_angle_correlation,
            'phase_duration_similarity': metrics.phase_duration_similarity,
            'rhythm_consistency': metrics.rhythm_consistency,
            'energy_correlation': metrics.energy_correlation,
            'dynamics_similarity': metrics.dynamics_similarity
        },
        'joint_range_similarity': metrics.joint_range_similarity,
        'detailed_feedback': metrics.detailed_feedback
    }

    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Create visualization if requested
    if args.visualize:
        viz_file = output_dir / f"comparison_{generated_path.stem}_{timestamp}.png"
        comparator.visualize_comparison(generated_path, reference_profile, metrics, viz_file)
        print(f"Visualization saved to: {viz_file}")


if __name__ == "__main__":
    # If no arguments, run a quick test with available files
    import sys

    if len(sys.argv) == 1:
        print("Running test comparison with example files...")

        # First, analyze and save profiles for reference BVH files
        analyzer = BVHReferenceAnalyzer()
        reference_files = [
            Path('bvh_examples/walk-through-spce.bvh'),
            Path('bvh_examples/walking-standing-still.bvh')
        ]

        for ref_file in reference_files:
            if ref_file.exists():
                print(f"\nAnalyzing reference: {ref_file.name}")
                profile = analyzer.analyze_reference_motion(ref_file)
                profile_path = ref_file.parent / f"{ref_file.stem}_profile.json"
                analyzer.save_profile(profile, profile_path)
                print(f"Saved profile: {profile_path}")

        # Test with a generated BVH if available
        test_files = [
            Path('bvh/walking_spine_fixed.bvh'),
            Path('bvh/walking_00001.bvh'),
            Path('bvh/thewave_spine_fixed.bvh')
        ]

        for test_file in test_files:
            if test_file.exists():
                print(f"\n{'='*60}")
                print(f"Testing: {test_file.name}")
                print('='*60)

                comparator = BVHReferenceComparator()

                # Use walk-through-spce as reference
                ref_profile = analyzer.analyze_reference_motion(reference_files[0])
                metrics = comparator.compare_to_reference(test_file, ref_profile)

                print(f"Grade: {metrics.quality_grade} ({metrics.overall_similarity:.1f}%)")
                print("\nKey Metrics:")
                print(f"  Gait Pattern: {metrics.gait_pattern_score:.1f}%")
                print(f"  Smoothness: {metrics.smoothness_comparison:.1f}%")
                print(f"  Joint Angles: {metrics.joint_angle_correlation:.1f}%")

                break
    else:
        main()