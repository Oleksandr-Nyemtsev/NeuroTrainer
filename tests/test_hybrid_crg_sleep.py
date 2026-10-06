import torch

from src.models.hybrid_crg_sleep import HybridCRGSleep


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    model = HybridCRGSleep().to(device)

    x = torch.randn(
        4,
        2,
        3000,
        device=device,
    )

    print("Input:", x.shape)

    with torch.no_grad():
        y = model(x)

    print("Output:", y.shape)

    assert y.shape == (4, 5)

    print("HYBRID CRG SLEEP TEST OK")


if __name__ == "__main__":
    main()

def test_hybrid_forward_cpu():
    model = HybridCRGSleep().cpu().eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 2, 3000))
    assert output.shape == (2, 5)
    assert torch.isfinite(output).all()
