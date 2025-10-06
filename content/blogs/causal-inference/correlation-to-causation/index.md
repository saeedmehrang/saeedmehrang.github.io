---
title: "The Quest for 'Why': Causal Inference, AGI, and the Limits of Pattern Recognition"
date: 2025-10-02
tags: [ "Causal Inference", "AGI", "Correlation vs Causation", "Causal AI"]
author: "Saeed Mehrang"
description: "An introduction to Causal Inference, exploring its historical roots, its distinction from mere correlation, and why understanding cause-and-effect is the key to achieving Artificial General Intelligence (AGI)."
summary: "This blog post introduces Causal Inference, highlighting the difference between correlation and causation and arguing that a true grasp of 'why' is necessary for the next evolution of AI."
cover:
    image: "quest-causation.png"
    alt: "Image representing cause and effect relationships or a question mark over data"
    relative: true
showToc: true
disableAnchoredHeadings: false

---

I have been exposed to **Causal Inference** for about 2 years now, with a much deeper exposure during 2024 after I joined **Vaalia Health** as a Lead Data Scientist. In this blog post I am writing an introduction and background on why we need to know about cause and effect relationships, and why we have not yet attained the true **AGI**. I will probably write more about this topic in the future, so please stay tuned for more content like this.

---

## Introduction

Imagine you are standing in a field thousands of years ago, and suddenly, the sky goes dark in the middle of the day. You might think a powerful spirit or god has swallowed the sun. This is how ancient people explained eclipses. They saw something strange and tried to find a reason for it, even if that reason was a myth. When they saw storms, floods, or diseases, they asked, “Why is this happening?” They noticed patterns — eclipses happening at certain times, floods after heavy rain — and tried to make a story about causes.

Even today, we ask similar questions. Science has advanced a lot, but we still see patterns and wonder about the real cause behind them. Does a special diet really help people lose weight, or do people who follow such diets already live a healthier lifestyle? Do hot days make people buy more ice cream, or is something else responsible? People have always tried to separate coincidence from real **cause-and-effect**. We don’t just want to see patterns; we want to know the deeper “why” behind them.

> “If one does not know to which port one is sailing, no wind is favorable.”
>
> — Seneca

This pursuit of understanding cause and effect is not just essential for human knowledge — it is also critical for **artificial intelligence**. The current generation of AI systems, including large language models, can recognize patterns and correlations in vast datasets, but they lack a deep understanding of causation. True **Artificial General Intelligence (AGI)**, a system capable of reasoning, adapting, and making autonomous decisions like a human, requires more than pattern recognition. It must comprehend and conform to cause-and-effect relationships, allowing it to distinguish between mere coincidences and genuine causal mechanisms. Without this ability, AI remains a tool for prediction rather than a reasoning entity capable of scientific discovery, decision-making, and ethical considerations.

Understanding causality is not just a philosophical or scientific pursuit — it is a necessary step toward developing machines that can think and reason in a human-like way. The ability to answer “why” questions will be the key to unlocking the next evolution of AI, bridging the gap between data-driven intelligence and truly autonomous systems.

---

## Historical and Philosophical Roots

Ancient philosophers wanted to go beyond myths to understand the world better. **Aristotle** for example talked about different kinds of causes — like what something is made of, how it is shaped, how it comes into existence, and what purpose it serves. He believed we don’t truly know something until we understand its cause.

> “We do not have knowledge of a thing until we have grasped its why, that is to say, its cause.”
>
> — Aristotle

Later, scientists like **Galileo** and **Newton** did experiments to check if things were really connected. They stopped relying on tradition and authority. Instead, they tested their ideas by observing and measuring. This showed that some beliefs, even those held for a long time, were just **correlations** and not real causes.

With scientific progress, governments and businesses also realized that understanding cause and effect could help them make better decisions. If they knew why certain policies worked, they could improve economies, health, and laws.

---

## The Shift in the Past Century

In the early 20th century, **Sir Francis Galton** and **Karl Pearson** made significant contributions to statistics by developing correlation measures. Pearson’s correlation coefficient became widely used to study relationships between variables. However, **correlation alone could not explain causation**. Scientists soon realized that relying purely on correlation was insufficient, especially in fields like medicine and economics.

As the limitations of correlation-based analysis became clearer, researchers began developing new ways to study causation. In the mid-20th century, statisticians like **Sewall Wright** introduced **path analysis**, a method that helped map out relationships between variables more clearly. Meanwhile, economists like **Trygve Haavelmo** laid the foundation for modern econometrics, which incorporated more structured causal models.

The field saw a major transformation with the work of **Judea Pearl**. In the 1980s and 1990s, Pearl introduced **causal diagrams (Directed Acyclic Graphs, or DAGs)** and the **do-calculus framework**, which provided a systematic way to distinguish correlation from causation. His work made it possible to mathematically define causal relationships and answer **counterfactual** questions — such as “What would have happened if we had done X instead of Y?”

Parallel to Pearl, in epidemiology and biostatistics, **James Robins** and **Miguel Hernán** advanced causal inference through the **g-methods**, including inverse probability weighting and marginal structural models. These methods allowed researchers to estimate causal effects in observational data where randomized experiments were not possible. **Susan Athey**, a leading economist, further advanced causal machine learning techniques, helping bridge the gap between causal inference and big data analytics.

Together, these pioneers have transformed causal inference from a philosophical question into a precise scientific discipline, shaping how we study medical treatments, policy decisions, and artificial intelligence.

Today, causal reasoning is not just important in science but in everyday life. Companies want to know if advertisements really increase sales. Health agencies study if a habit improves health or if people who already have better health are just more likely to follow that habit. The better we understand cause and effect, the better we can shape our world.

---

## The Difference Between Correlation and Causation

We see patterns all the time. When the weather gets hot, people wear lighter clothes. When people go swimming, ice cream sales go up. But this does not mean one thing causes the other.

A common example is ice cream sales and heatstroke. On hot days, both ice cream sales and cases of heatstroke increase. But that does not mean eating ice cream causes heatstroke. The real cause is the hot weather.

Not understanding this difference can lead to mistakes. A financial analyst might see that a company’s stock rises whenever another market index goes up and assume that one causes the other. Or a government official might think that more parks in an area directly lead to better health when, in fact, richer areas invest in many things that improve health.

Getting this wrong can lead to bad decisions. In medicine, doctors must be careful not to give treatments just because they appear to help without proper studies proving they actually work. Businesses that follow misleading data might invest in the wrong products or strategies.

---

## Challenges in Discovering Causation

The real world is complicated. Finding real causes is much harder than seeing patterns. Systems in health, economics, and the environment have many connected parts. It can be difficult to tell which factor is truly responsible for an outcome.

Another problem is that we can’t always do experiments to test causes. In medicine, scientists use **randomized controlled trials (RCTs)** to test if a new drug really helps. But in public policy or the environment, we can’t always test things this way.

> “Nothing ever is, everything is becoming.”
>
> — Heraclitus

To solve this, scientists have developed new methods to map and study cause-and-effect relationships. One useful tool is the **Directed Acyclic Graph (DAG)**, a visual representation of causal structures. A DAG consists of **nodes** (representing variables) and **directed edges** (arrows) that indicate causal relationships between them. DAGs help researchers identify confounding variables, determine whether an observed relationship is spurious, and clarify whether an intervention on one variable can meaningfully influence another. By systematically applying rules to a DAG, scientists can decide whether a given dataset can answer a causal question or if additional data is needed.

### Markov Equivalence and Its Implications

A key challenge in causal discovery is **Markov Equivalence**, which refers to the fact that multiple DAGs can imply the same statistical relationships among variables. That means different causal structures can generate identical data distributions, making it impossible to determine the correct causal model based on observational data alone. To resolve this, researchers use additional domain knowledge or interventions (such as experiments) to distinguish among equivalent structures. Understanding Markov Equivalence helps scientists recognize the limits of purely observational studies and the necessity of well-designed experiments or assumptions to uncover causal mechanisms.

### Causality in Time-Varying Settings

Many real-world situations involve variables that change over time. For example, in healthcare, a patient’s treatment decisions evolve as their condition changes. This makes causal inference more complex, as past exposures can influence future treatments, which in turn affect later outcomes. These situations require special techniques, **g methods**, to properly estimate causal effects. Without accounting for time-varying confounders, standard statistical methods can produce misleading results. DAGs can be extended to **Dynamic DAGs (D-DAGs)**, which allow researchers to represent and analyze causality in such evolving systems.

> “Reason is immortal, all else mortal.”
>
> — Pythagoras

---

## Modern Applications of Causal Inference (2025 and Beyond)

As data-driven decision-making becomes more prominent, causal inference is transforming industries and fields.

Here are some key examples of how causal methods are shaping the world in 2025:

* **Causal Inference in Personalized Medicine**: Healthcare systems in 2025 leverage vast amounts of patient data. Traditional machine learning tools can identify patterns, but causal models determine which treatments genuinely improve patient outcomes rather than merely coinciding with recovery.
* **Tackling Climate Change Interventions**: Researchers use causal inference to determine which climate policies genuinely reduce emissions rather than just appearing alongside improvements. This ensures that investments are made in interventions with a proven impact.
* **Optimizing Online Education**: EdTech platforms use causal inference to identify whether instructional videos truly enhance learning outcomes, or if students who watch them were already more engaged. This enables educators to design more effective courses.
* **Streamlined Manufacturing and Logistics**: Companies apply causal diagrams to diagnose supply chain bottlenecks, distinguishing between true causes of delays versus external economic trends.
* **Fairness in Algorithmic Decision-Making**: AI-driven hiring systems and lending algorithms are scrutinized using causal inference to detect and correct hidden biases, ensuring fairer and more ethical decision-making.
* **More Intelligent Virtual Assistants**: By incorporating causal inference, digital assistants move beyond surface-level recommendations and provide users with more realistic “if-then” scenarios, making them truly useful in decision-making.

---

## Final Words

From stories about gods and eclipses to modern statistical methods, humanity’s longing to answer “Why?” has driven countless innovations. The distinction between correlation and causation is not just a technical point — it can shape real-world decisions in health, policy, and commerce. Our best strategies for uncovering true causes involve rigorous tests and structured frameworks that keep us from mixing up coincidental patterns with genuine effects.

As artificial intelligence evolves, we may soon see machines that not only recognize patterns but also reason about cause and effect. The quest for “why” is as old as humanity itself, and now, with AI and causal reasoning, we may finally uncover answers that have remained elusive for centuries.

---

## Current and Future Trends

I personally believe that we are gonna see a wave of new techniques and advancements in **Causal Discovery**, especially with the help of advanced reasoning LLMs such as DeepSeek R1 and OpenAI O series models. For some reason, there has been a delay of adopting more sophisticated deep learning based techniques for advancing Causal Inference. However, I can see that the change has started and we are seeing some interesting and practical innovations.

---

## References

* What If (Free Online Book): https://miguelhernan.org/whatifbook
* The Book of Why: https://www.amazon.com/Book-Why-Science-Cause-Effect/dp/046509760X
* Course: https://www.edx.org/learn/data-analysis/harvard-university-causal-diagrams-draw-your-assumptions-before-your-conclusions