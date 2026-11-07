# ThinkBooster — LinkedIn post

Ready to paste. The first two lines are the hook (LinkedIn hides the rest behind "see more").
Fill in the Medium URL once it's published; optionally @-tag your co-authors and MBZUAI / ETH Zürich.

---

ThinkBooster was accepted to the ACL 2026 System Demonstration Track.

It turns any LLM into its own "Pro reasoning mode" by changing one URL, and it finally tells you what the extra accuracy costs.

Every frontier model can "think longer" at inference now. But there are many ways to spend that test-time compute: sample and rerank, search a tree of reasoning steps, stop when the model is confident, add compute only on the hard steps. They cost wildly different amounts, and almost no one reports the cost. So the question that actually matters, "for the accuracy I need, which method is cheapest?", has had no clean answer.

ThinkBooster is built to answer it. It is an open-source framework that:

• puts 9 test-time scaling strategies and 4 scorer families behind one API
• measures accuracy and compute (TFLOPs + tokens) together, on bundled math, coding, and science benchmarks
• ships as an OpenAI-compatible proxy: point your client at /v1/<strategy>/<scorer> and the same model gets test-time scaling, with no change to your app
• includes a visual debugger that shows why a reasoning trajectory was selected

Two results that cut against intuition:

→ On HumanEval+, a free uncertainty signal beat a trained reward model, taking Qwen3-8B from 79.3 to 88.8.
→ Dropping best-of-N with a reward model into GPT-OSS-120B raised its CUDA-kernel correctness from 26 to 30.

Built by MBZUAI, ETH Zürich, Imperial College London, NUS, and collaborators.

📄 Paper (arXiv): https://arxiv.org/abs/2606.06915
📝 Deep dive (Medium): [add Medium URL once published]
🕹️ Live demo: http://demo-thinkbooster.nlpresearch.group
💻 Code (GitHub): https://github.com/IINemo/thinkbooster

#LLM #AI #NLP #MachineLearning #ACL2026 #Reasoning #OpenSource

---

## Notes / before posting

- **Medium URL**: replace the placeholder once the article is live.
- **Tag people**: on LinkedIn, @-mention co-authors and MBZUAI / ETH Zürich / Imperial / NUS for reach.
- **Emoji**: the link icons (📄 📝 🕹️ 💻) and bullets are standard LinkedIn formatting; strip them if you want it fully plain.
- **Demo link is http-only** (ephemeral pod) — keep the pod up while the post is circulating, or swap to the GitHub/paper links if it's down.
- **First comment trick**: LinkedIn down-ranks posts with outbound links. Consider moving the four links into the first *comment* and ending the post body with "Links in the comments." for better reach.
- **Length**: ~280 words. Fine for LinkedIn; trim the feature bullets to 2 if you want it shorter.
