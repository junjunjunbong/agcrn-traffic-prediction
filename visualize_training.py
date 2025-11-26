"""
학습 history 시각화 스크립트
Training history를 읽어서 손실 변화 그래프를 생성합니다.
"""
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.config import LOGS_DIR


def load_training_history(history_path: Path = None):
    """
    학습 history JSON 파일을 로드합니다.

    Args:
        history_path: history 파일 경로 (기본값: logs/training_history.json)

    Returns:
        history: 학습 history 딕셔너리
    """
    if history_path is None:
        history_path = LOGS_DIR / "training_history.json"

    if not history_path.exists():
        raise FileNotFoundError(f"Training history file not found: {history_path}")

    with open(history_path, 'r') as f:
        history = json.load(f)

    return history


def plot_training_history(history, save_path: Path = None, show: bool = True):
    """
    학습 손실 변화를 시각화합니다.

    Args:
        history: 학습 history 딕셔너리
        save_path: 그래프 저장 경로
        show: 그래프를 화면에 표시할지 여부
    """
    train_losses = history.get('train_losses', [])
    val_losses = history.get('val_losses', [])
    best_val_loss = history.get('best_val_loss', None)

    if not train_losses:
        print("⚠️  No training data found in history")
        return

    epochs = range(1, len(train_losses) + 1)

    # 그래프 생성
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 1. Train vs Validation Loss
    ax1 = axes[0]
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.8)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2, alpha=0.8)

    # Best validation loss 표시
    if best_val_loss is not None:
        best_epoch = val_losses.index(min(val_losses)) + 1
        ax1.axhline(y=best_val_loss, color='g', linestyle='--',
                    label=f'Best Val Loss: {best_val_loss:.6f}', alpha=0.7)
        ax1.scatter([best_epoch], [best_val_loss], color='g', s=100, zorder=5)
        ax1.annotate(f'Epoch {best_epoch}',
                     xy=(best_epoch, best_val_loss),
                     xytext=(10, 10), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. Loss 감소율 (로그 스케일)
    ax2 = axes[1]
    ax2.semilogy(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.8)
    ax2.semilogy(epochs, val_losses, 'r-', label='Val Loss', linewidth=2, alpha=0.8)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss (log scale)', fontsize=12)
    ax2.set_title('Loss Curve (Log Scale)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # 저장
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 그래프 저장 완료: {save_path}")

    # 화면 표시
    if show:
        plt.show()
    else:
        plt.close()

    # 통계 정보 출력
    print("\n" + "="*60)
    print("학습 통계")
    print("="*60)
    print(f"총 에포크 수: {len(train_losses)}")
    print(f"최종 Train Loss: {train_losses[-1]:.6f}")
    print(f"최종 Val Loss: {val_losses[-1]:.6f}")
    if best_val_loss is not None:
        print(f"최고 Val Loss: {best_val_loss:.6f} (Epoch {best_epoch})")
    print(f"Train Loss 감소: {train_losses[0]:.6f} → {train_losses[-1]:.6f} "
          f"({(1 - train_losses[-1]/train_losses[0])*100:.2f}% 감소)")
    print(f"Val Loss 감소: {val_losses[0]:.6f} → {val_losses[-1]:.6f} "
          f"({(1 - val_losses[-1]/val_losses[0])*100:.2f}% 감소)")
    print("="*60)


def plot_loss_comparison(history_files: list, labels: list = None,
                         save_path: Path = None, show: bool = True):
    """
    여러 학습 history를 비교 시각화합니다.

    Args:
        history_files: history 파일 경로 리스트
        labels: 각 history의 라벨 (기본값: 파일명)
        save_path: 그래프 저장 경로
        show: 그래프를 화면에 표시할지 여부
    """
    if labels is None:
        labels = [Path(f).stem for f in history_files]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(history_files)))

    for i, (history_file, label) in enumerate(zip(history_files, labels)):
        with open(history_file, 'r') as f:
            history = json.load(f)

        train_losses = history.get('train_losses', [])
        val_losses = history.get('val_losses', [])
        epochs = range(1, len(train_losses) + 1)

        color = colors[i]
        ax1.plot(epochs, train_losses, color=color, linestyle='-',
                 label=f'{label} (Train)', linewidth=2, alpha=0.7)
        ax2.plot(epochs, val_losses, color=color, linestyle='-',
                 label=f'{label} (Val)', linewidth=2, alpha=0.7)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Train Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 비교 그래프 저장 완료: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='학습 history를 시각화합니다',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 사용 (logs/training_history.json 시각화)
  python visualize_training.py

  # 특정 history 파일 시각화
  python visualize_training.py --history logs/experiment1.json

  # 그래프 저장
  python visualize_training.py --save results/training_plot.png

  # 여러 history 비교
  python visualize_training.py --compare logs/exp1.json logs/exp2.json --labels "실험1" "실험2"

  # 화면에 표시하지 않고 저장만
  python visualize_training.py --save plot.png --no-show
        """
    )

    parser.add_argument('--history', type=str, default=None,
                        help='학습 history JSON 파일 경로 (기본값: logs/training_history.json)')
    parser.add_argument('--save', type=str, default=None,
                        help='그래프를 저장할 경로')
    parser.add_argument('--no-show', action='store_true',
                        help='그래프를 화면에 표시하지 않음')
    parser.add_argument('--compare', nargs='+', type=str,
                        help='비교할 여러 history 파일들')
    parser.add_argument('--labels', nargs='+', type=str,
                        help='비교 시 각 history의 라벨')

    args = parser.parse_args()

    print("="*60)
    print("학습 History 시각화")
    print("="*60)

    try:
        if args.compare:
            # 여러 history 비교
            print(f"비교할 파일 수: {len(args.compare)}")
            for f in args.compare:
                print(f"  - {f}")

            save_path = Path(args.save) if args.save else None
            plot_loss_comparison(
                history_files=args.compare,
                labels=args.labels,
                save_path=save_path,
                show=not args.no_show
            )
        else:
            # 단일 history 시각화
            if args.history:
                history_path = Path(args.history)
                print(f"History 파일: {history_path}")
            else:
                history_path = LOGS_DIR / "training_history.json"
                print(f"History 파일: {history_path} (기본값)")

            history = load_training_history(history_path)

            save_path = Path(args.save) if args.save else None
            plot_training_history(
                history=history,
                save_path=save_path,
                show=not args.no_show
            )

        print("\n✓ 시각화 완료!")

    except FileNotFoundError as e:
        print(f"\n❌ 오류: {e}")
        print("\n💡 먼저 모델을 학습시켜야 합니다:")
        print("   python train.py")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        raise


if __name__ == "__main__":
    main()
