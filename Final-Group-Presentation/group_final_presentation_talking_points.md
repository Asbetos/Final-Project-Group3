# Group Final Presentation Talking Points

## Slide 1: Title

### Talk Track
- Our project studies speculative decoding as a practical way to accelerate large language model inference.
- To explain the motivation, it helps to first describe how regular LLM inference works.
- In a normal autoregressive LLM, we give the model an input prompt, the model converts it into tokens, runs those tokens through the full transformer stack, produces a probability distribution over the next token, samples or selects one token, appends it to the sequence, and then repeats that same process again for the next token.
- So even though the model has already seen the previous context, generation still happens one token at a time.
- That is slow because every new token requires another pass through a very large target model, and those repeated passes become expensive in latency, memory bandwidth, and total wall-clock time.
- Our project improves that by adding a draft component, either a separate small draft model or a learned EAGLE-3 draft head, that proposes several likely future tokens before the large target model verifies them.
- If the target agrees with those proposals, generation can advance by multiple tokens in one verification step instead of only one token per full target-model pass.
- More specifically, we benchmarked two acceleration paths for Gemma on mid-tier GPU hardware: standard speculative decoding with a separate draft model, and EAGLE-3 with a learned draft head.
- Our final results focus on what actually completed successfully and reproducibly on an NVIDIA A10G instance.

### Anticipated Questions
- Q: What is the one-sentence summary of the project?
  A: We tested when speculative decoding actually speeds up Gemma inference on realistic hardware, and compared a separate draft model against a learned EAGLE-3 draft head.
- Q: How does regular LLM inference work in one sentence?
  A: The model predicts one next token at a time, appends it to the context, and repeats that process sequentially until generation stops.
- Q: Why is that slow?
  A: Because every token requires another expensive forward pass through a large model, so latency accumulates token by token.
- Q: What is the project's core improvement over standard inference?
  A: We try to move from one-token-per-pass generation to multi-token advancement by drafting likely continuations and verifying them with the target model.
- Q: Why Gemma?
  A: Gemma gave us an open, modern model family with practical 12B-scale targets that fit the project hardware and let us study both standard and EAGLE-style acceleration.
- Q: Why emphasize mid-tier GPU hardware?
  A: Many papers report results on flagship GPUs, but many real deployments use cards like the A10G, where memory and latency constraints matter much more.

## Slide 2: Problem Selection

### Talk Track
- We chose this problem because autoregressive generation is fundamentally sequential: the model generates one token at a time, and each token requires another decoding pass.
- That means inference often becomes the bottleneck, especially for longer generations.
- So the systems question is not just whether the model is good, but whether we can get the same output distribution faster.
- Speculative decoding is attractive because it directly targets this bottleneck without retraining the target model itself.
- It also connects core NLP ideas like autoregressive modeling, next-token probabilities, and sampling, with systems concerns like throughput and memory bandwidth.
- The before-and-after picture on this slide captures the practical goal: move from one token per pass to several tokens per verification step.

### Anticipated Questions
- Q: Why is inference bottlenecked if the model already fits in memory?
  A: Because generation is still sequential, so even when memory is sufficient, the model has to do repeated decoding passes and that limits throughput.
- Q: Does speculative decoding change output quality?
  A: In theory, no, because the target model still verifies or corrects the draft proposals, so the final distribution is preserved.
- Q: Why is this an NLP problem and not just a systems problem?
  A: Because the speedup depends directly on language-model behavior, especially how predictable the next-token distribution is for different tasks.

## Slide 3: How Standard Speculative Decoding Works

### Talk Track
- This slide gives the core mechanics of standard speculative decoding.
- First, both models start from the same input context.
- Second, the draft model proposes gamma candidate tokens.
- Third, the target model evaluates that proposal in one batched verification step.
- Fourth, accepted tokens are appended, rejected tokens are corrected, and the process repeats.
- The most important control variable here is gamma, because it determines how many draft tokens we try to gain before paying for target verification.
- If gamma is too small, we do not gain much. If it is too large and the draft is poorly aligned, we waste work on rejected tokens.

### Anticipated Questions
- Q: What exactly is gamma?
  A: Gamma is the speculation length, meaning the number of draft tokens proposed before the target model verifies them.
- Q: Why can the target verify multiple tokens in one step?
  A: Because it can score the proposed continuation in a batched forward pass rather than recomputing from scratch token by token.
- Q: What makes speculative decoding fail?
  A: Low draft-target agreement. If too many proposed tokens are rejected, the extra draft work outweighs the benefit.

## Slide 4: Final Gemma Experimental Scope

### Talk Track
- This slide is important because it explains our final scope.
- We began with a broader search across possible model pairings, but the final report intentionally narrows to the completed and reliable Gemma-3-12B configurations.
- Module 1 is pair F: standard speculative decoding with Gemma-3-12B as the target and Gemma-3-1B as the separate draft model.
- Module 2 is pair I: EAGLE-3, where Gemma-3-12B remains the target but the draft component is a trained draft head rather than a second full language model.
- It is also important to clarify that both methods are still sequential in an autoregressive sense.
- In standard speculative decoding, the small draft model does not generate all future tokens in one truly parallel pass. It still generates them sequentially, but because it is much smaller than the target model, those draft steps are cheap.
- The speedup comes from the fact that the large target model can verify an entire drafted chunk in one batched step, so we reduce the number of expensive target-model passes.
- EAGLE-3 follows the same high-level propose-and-verify pattern, but the proposal mechanism is different.
- Instead of using a second full draft model, EAGLE-3 uses a lightweight trained draft head on top of the target model hidden states.
- So a good way to compare the two modules is this: pair F uses a separate small model to draft ahead, while EAGLE-3 uses a lightweight internal prediction head attached to the target model itself.
- If I explain it intuitively, standard speculation is like using a fast junior engineer to draft ahead, while EAGLE-3 is like giving the senior engineer a very fast internal autocomplete tool.
- Both final configurations stayed under ten gigabytes of active inference VRAM, which made them practical on an AWS g5.2xlarge with an NVIDIA A10G.
- The scope shift matters because it turns the project into a clean comparison between two acceleration philosophies: separate draft model versus learned draft head.

### Anticipated Questions
- Q: Why did you drop other configurations from the final report?
  A: We wanted the final story to focus on completed, reproducible experiments instead of partially explored branches.
- Q: Why is pair I a meaningful comparison to pair F?
  A: Both use the same 12B target, so the main difference is the acceleration mechanism.
- Q: Can the draft model really generate multiple tokens in parallel?
  A: Not truly in parallel. It still generates them sequentially, but it is much smaller, so drafting is cheap. The parallelism benefit comes from target-model verification.
- Q: Is EAGLE-3 also sequential?
  A: Yes, in the sense that it still predicts future tokens conditioned on the current context. The difference is that it uses a lightweight draft head over target hidden states instead of a second full model.
- Q: Then where does the speedup come from in both methods?
  A: In both cases, the gain comes from reducing how often the large target model has to do expensive decoding passes.
- Q: What is the simplest way to explain the difference between the two modules?
  A: Pair F drafts with a separate small language model, while EAGLE-3 drafts with a trained internal head built on the target model hidden states.
- Q: What does under 10 GB buy you?
  A: It means these methods are deployable on practical mid-tier GPU hardware instead of requiring a flagship accelerator.

## Slide 5: Datasets

### Talk Track
- We used four task domains because speculative decoding performance depends strongly on output predictability.
- HumanEval represents highly structured code completion.
- TriviaQA represents short factual question answering.
- CNN/DailyMail represents long-context summarization.
- WritingPrompts represents higher-entropy creative generation.
- Each configuration used fifty prompts with a fixed seed of forty-two.
- The dataset mix is deliberate: it lets us test whether acceleration works better for structured outputs than for open-ended outputs.

### Anticipated Questions
- Q: Why only 50 prompts per task?
  A: It was a practical tradeoff between runtime cost and comparative coverage, especially because the full experiment grid was large.
- Q: Why these four tasks?
  A: They span a useful range from highly constrained to high-entropy generation, which is exactly what we needed to study acceptance behavior.
- Q: Did you use the same prompt pipeline across modules?
  A: Yes. Both acceleration modules evaluated on the same four benchmark task families.

## Slide 6: Module 1 Experimental Setup

### Talk Track
- This slide is the experimental setup for the standard speculative-decoding module, and this is one of the slides I would emphasize.
- The active configuration is pair F: Gemma-3-12B-it as the target and Gemma-3-1B-it as the draft, with a 4-bit target and a BF16 draft.
- We varied gamma over 1, 3, 5, 7, and 10.
- We varied temperature over 0.0, 0.6, and 1.0.
- We evaluated all four tasks, used fifty prompts per task, and limited generation to 128 new tokens.
- For each configuration, we first ran a baseline with the target model only, then ran speculative decoding with the same task and prompt settings.
- We recorded acceptance rate, accepted length, tokens per second, speedup, time to first token, peak VRAM, and draft overhead.
- The purpose of this setup was to identify when the speed gained from accepted draft tokens is large enough to outweigh the cost of running the draft model.

### Anticipated Questions
- Q: Why did you search gamma over those five values?
  A: They gave us a meaningful range from conservative to aggressive drafting without making the grid unmanageably large.
- Q: Why include multiple temperatures?
  A: Because randomness changes the next-token distribution and therefore changes draft-target agreement.
- Q: Why run a baseline for every task and temperature?
  A: Speedup is only meaningful relative to a matched baseline under the same task and decoding conditions.
- Q: Why was the target 4-bit quantized?
  A: To keep the 12B target practical on the A10G while preserving the overall generation workflow.

## Slide 7: Pair F Results

### Talk Track
- The high-level result is that pair F helps selectively rather than universally.
- HumanEval and TriviaQA exceed baseline, while CNN/DailyMail and WritingPrompts remain below baseline.
- The best HumanEval result reached 1.476x, and the best TriviaQA result reached 1.295x.
- In contrast, the summarization and creative-writing tasks stayed below 1.0x, meaning the speculative path was slower than the baseline there.
- So the main takeaway is that structured tasks benefited, but open-ended tasks did not.

### Anticipated Questions
- Q: What was the strongest pair F result?
  A: HumanEval at 1.476x speedup.
- Q: Why did WritingPrompts stay below baseline?
  A: Because the output distribution is less predictable, so the draft model proposals are less likely to be accepted.
- Q: Does that mean pair F is bad?
  A: No. It means it is task-dependent, which is exactly the kind of practical conclusion we wanted.

## Slide 8: Pair F Acceptance Explains Speedup

### Talk Track
- This slide explains why the previous slide looks the way it does.
- HumanEval and TriviaQA maintained higher acceptance rates.
- CNN/DailyMail and WritingPrompts remained substantially lower.
- That matters because speculative decoding only wins when the target can keep enough of the draft model proposed tokens.
- If acceptance is low, then the system pays the extra draft-model cost but still has to redo too much work at target-verification time.
- So acceptance rate is the central systems variable behind the speedup pattern.

### Anticipated Questions
- Q: Is acceptance rate the most important metric?
  A: It is one of the most important explanatory metrics because it directly tells us how aligned the draft is with the target.
- Q: Why is HumanEval acceptance so high?
  A: Code completion is more structured, so the next-token distribution is narrower and easier for the draft model to predict.
- Q: Could a better draft model fix the low-acceptance tasks?
  A: Possibly, but the cost of the draft model and the complexity of the task both matter.

## Slide 9: Pair F Throughput-Latency Trade-off

### Talk Track
- This slide shows that throughput and latency are not the same thing.
- Pair F can improve sustained throughput on structured tasks, but the best speedup settings also increase TTFT, which is time to first token.
- So a configuration can be faster in tokens per second while still feeling slower at the start of generation.
- This is important for deployment because some applications care most about total throughput, while others care most about responsiveness.
- Our practical conclusion is that standard speculative decoding is more attractive for longer sustained generations than for low-latency first-token use cases.

### Anticipated Questions
- Q: Why does TTFT go up when throughput improves?
  A: Because the system spends more work on drafting and verification up front before the first visible token is finalized.
- Q: Which metric matters more?
  A: It depends on the application. Batch or long-form generation cares more about throughput, while interactive chat often cares more about TTFT.
- Q: Did pair F stay practical in memory?
  A: Yes, peak VRAM stayed below about ten gigabytes.

## Slide 10: Module 2 Experimental Setup, EAGLE-3

### Talk Track
- This is the experimental setup for Module 2, and this is the other setup slide I would emphasize.
- Here we replace the separate draft model with a trained EAGLE-3 draft head built on top of the target model hidden states.
- The active configuration is pair I: Gemma-3-12B-it as a 4-bit frozen target, plus a trained EAGLE-3 draft head.
- The objective is to predict future tokens from internal target-model features instead of from a second language model.
- The draft head uses three target hidden layers, fusion and input projections, one copied decoder layer, and the frozen target norm and LM head.
- In the inference sweep, the main search variable is tree budget rather than gamma, and we evaluated tree budgets 20 and 60 across the same task set and temperatures.
- Conceptually, this is a more compact acceleration path because the draft component is learned directly around the target model instead of being a separate external model.

### Anticipated Questions
- Q: What is the main conceptual difference from pair F?
  A: Pair F uses a second full draft model, while EAGLE-3 uses a learned draft head over the target model hidden states.
- Q: Why is that appealing?
  A: It can reduce the footprint of the acceleration component while keeping the proposal process closely tied to the target.
- Q: What is tree budget?
  A: It controls how many candidate draft nodes are explored before verification.
- Q: Did you still keep the target model frozen?
  A: Yes. The target remained frozen and the trainable part was the draft head.

## Slide 11: Pair I Training Results

### Talk Track
- The most important message from this slide is that the EAGLE-3 training run completed successfully.
- The loss dropped from 145.3 at step 10 to below 1.0 by the final epoch.
- The run used 52,002 prepared training examples and produced a usable final checkpoint.
- The draft head had about 297.9 million trainable parameters and used an 8-bit AdamW optimizer.
- So by the end of training, we had a real learned acceleration component rather than just a conceptual design.

### Anticipated Questions
- Q: How do you know the head learned something useful?
  A: The training loss dropped dramatically and the final checkpoint was good enough to produce selective inference gains downstream.
- Q: Why train on Alpaca-GPT4 if evaluation is on other tasks?
  A: Because the goal is to learn a general hidden-state extrapolator, then test it on held-out task families.
- Q: How long did training take?
  A: The completed run took roughly thirty-seven active training hours on the A10G.

## Slide 12: Pair I Results

### Talk Track
- The EAGLE-3 inference results are even more selective than pair F.
- TriviaQA is the clear success case, peaking at 2.368x speedup with tree budget 20 and temperature 0.0.
- WritingPrompts comes close to baseline but does not clearly exceed it.
- HumanEval and CNN/DailyMail remain below baseline.
- CNN/DailyMail is especially difficult, which suggests that long-context verification is still a limitation in the current implementation.
- Another practical observation is that tree budget 20 outperformed tree budget 60 in the best final outcomes, so larger trees did not help once verification cost was included.

### Anticipated Questions
- Q: Why is TriviaQA so strong for EAGLE-3?
  A: Because it is factual, short-form, and more predictable, so the draft head proposed paths are accepted much more often.
- Q: Why is CNN/DailyMail so weak?
  A: It is long-context summarization, which is harder for hidden-state extrapolation and also stresses verification and cache behavior.
- Q: Is EAGLE-3 better than standard speculation overall?
  A: Not universally, but its best result is better, because 2.368x on TriviaQA is the strongest acceleration result in the project.

## Slide 13: Conclusion

### Talk Track
- Our final conclusion is that both pair F and EAGLE-3 are viable acceleration paths, but both are highly task-dependent.
- Speedup depends primarily on draft-target agreement.
- When acceptance is high, structured tasks can show strong gains.
- When acceptance is low, the extra draft or verification work can erase the benefit completely.
- A practical deployment decision therefore has to balance throughput, TTFT, VRAM, and engineering complexity.
- Our clearest next step is to improve long-context EAGLE verification, especially for summarization-style tasks like CNN/DailyMail.

### Anticipated Questions
- Q: What is the single biggest lesson?
  A: Speculative decoding is not automatically faster; it is faster only when the proposal mechanism is well aligned with the task.
- Q: What is the best result of the whole project?
  A: EAGLE-3 on TriviaQA at 2.368x speedup.
- Q: What would you improve first?
  A: Long-context EAGLE verification, because that is where the largest remaining weakness is.

## Slide 14: Thank You

### Talk Track
- Thank you. We would be happy to take questions.
- If useful, we can discuss the trade-off between standard speculative decoding and EAGLE-3, the role of acceptance rate, or the constraints imposed by A10G-class hardware.

### Anticipated Questions
- Q: If you had more time, what would you add?
  A: A larger validation study for EAGLE-3 training and more long-context engineering for summarization tasks.
- Q: What should the audience remember most?
  A: Acceleration works best when the proposal mechanism matches the structure of the task.

## Delivery Tips

- Spend the most time on Slides 2, 3, 4, 6, and 10.
- On Slides 7 through 12, keep repeating the same logic: acceptance explains speedup.
- If time is short, compress Slides 8 and 9 into one sentence each, but do not skip Slides 6 and 10.
- When discussing results, always mention the best concrete numbers: 1.476x for Pair F on HumanEval and 2.368x for EAGLE-3 on TriviaQA.
- If asked about failures, frame them positively: the project still identified exactly when these accelerators help and when they do not.
