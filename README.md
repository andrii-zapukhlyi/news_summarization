# Reducing News Reading Time with Transformer-Based Summaries

## Objective
In today’s fast-paced world, a constant stream of news makes it difficult for people to stay informed—especially about political events—without spending a lot of time reading full articles. The goal is to help users quickly grasp key information without sacrificing understanding.

## Solution
To address this issue, I will develop an automatic news summarization system that generates concise and informative summaries using Deep Learning models. This project focuses on comparing standard neural networks, including those with Attention Mechanisms, to state-of-the-art Transformer models. Additionally, it aims to train a smaller model for experimental purposes, exploring the performance differences in NLP tasks.

## Achievements  
The T5-Small model generated the following summary for the "GAME OF CHICKEN" section in the article <a href = "https://www.reuters.com/world/trumps-latest-tariffs-loom-set-deepen-global-trade-war-2025-04-09/">China retaliates as Trump trade tariffs kick in</a>:
```
China's move to impose 84% retaliatory tariffs is a losing proposition for Beijing. U.S. Treasury Secretary Scott Bessent: "They are the worst offenders in the international trading system" Trump nearly doubled duties on Chinese imports, which had been set at 54% last week.
```

The BART-Large-CNN model generated this summary for the same text:
```
The U.S. and China are in an "unprecedented and expensive game of chicken," says CNN's John Sutter. Sutter: "It's unfortunate that the Chinese are the worst offenders in the international trading system"
```

### ROUGE Scores:
- **T5**: ROUGE-1 = 0.39, ROUGE-2 = 0.37, ROUGE-L = 0.39
- **BART**: ROUGE-1 = 0.27, ROUGE-2 = 0.23, ROUGE-L = 0.17

```
ROUGE measures word overlap. If two texts express the same meaning but use different wording, the ROUGE score may still be low. Therefore, it's crucial for summaries to be evaluated by a human to ensure they are genuinely meaningful.
```

## Conclusion
This project successfully developed a news summarization system using Deep Learning models. By training the T5-Small model with 60 million parameters, I achieved results comparable to the larger BART-Large-CNN model with 400 million parameters. Additionally, I explored LSTM and LSTM with attention models to compare traditional and Transformer-based architectures. The results demonstrate that Transformer models can generate concise, informative summaries, reducing reading time without losing key content.