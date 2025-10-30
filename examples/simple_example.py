from rewardsignal import SignalClient
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")


def main():
    client = SignalClient(api_key=API_KEY)
    run = client.create_run(
        base_model="meta-llama/Llama-3.2-3B",
        lora_r=32,
        lora_alpha=64,
        learning_rate=3e-4,
    )

    # Example PPO training batch (reward modeling pairs)
    ppo_batch = [
        {
            "prompt": "What is the capital of France?",
            "chosen": "Paris is the capital of France.",
            "rejected": "London is the capital of France.",
        },
        {
            "prompt": "Explain quantum computing in simple terms.",
            "chosen": "Quantum computing uses quantum mechanics to process information, allowing some problems to be solved faster than with classical computers.",
            "rejected": "Quantum computing is just a faster version of normal computing.",
        },
        {
            "prompt": "Who wrote the play Romeo and Juliet?",
            "chosen": "William Shakespeare wrote Romeo and Juliet.",
            "rejected": "It was written by Charles Dickens.",
        },
    ]

    training_batch = ppo_batch

    for step in range(20):
        fb_result = run.forward_backward(batch=training_batch)
        loss = fb_result["loss"]
        grad_norm = fb_result.get("grad_norm", 0.0)
        print(f"Step {step:3d} | Loss: {loss:.4f} | Grad Norm: {grad_norm:.4f}")
        run.optim_step()
        if step % 5 == 0:
            _ = run.sample(
                prompts=["The meaning of life is"],
                max_tokens=50,
                temperature=0.7,
                top_p=0.9,
            )

    _ = run.save_state(mode="adapter")

    # Optionally push to HuggingFace Hub
    # hub_result = run.save_state(
    #     mode="adapter",
    #     push_to_hub=True,
    #     hub_model_id="your-username/your-model-name"
    # )

    # Gradient Accumulation
    for step in range(20):
        fb_result = run.forward_backward(batch=training_batch, accumulate=True)
        loss = fb_result["loss"]
        grad_norm = fb_result.get("grad_norm", 0.0)
        print(f"Step {step:3d} | Loss: {loss:.4f} | Grad Norm: {grad_norm:.4f}")
        run.optim_step()
        if step % 5 == 0:
            _ = run.sample(
                prompts=["The meaning of life is"],
                max_tokens=50,
                temperature=0.7,
                top_p=0.9,
            )

    # Learning Rate Schedule
    for step in range(20):
        fb_result = run.forward_backward(batch=training_batch)
        loss = fb_result["loss"]
        grad_norm = fb_result.get("grad_norm", 0.0)
        print(f"Step {step:3d} | Loss: {loss:.4f} | Grad Norm: {grad_norm:.4f}")
        run.optim_step()
        if step % 5 == 0:
            _ = run.sample(
                prompts=["The meaning of life is"],
                max_tokens=50,
                temperature=0.7,
                top_p=0.9,
            )


if __name__ == "__main__":
    print("Signal API - Supervised Fine-Tuning Example")

    # Run main example
    main()
