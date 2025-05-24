**Introduction**

This book originates from notes I wrote for two university courses. The first is ORIE5250: Topics in Risk Management and Portfolio Construction, a course offered in the program for M.S. in Financial Engineering at Cornell University. The second is MATH-GA2011: Algorithmic Trading & Quantitative Strategies, offered in the program for M.S. in Mathematics in Finance at NYU Courant. Since this book’s objective was to write the quantitative finance book I had wanted to read at the beginning of my journey in finance. Given the scope and goals of quantitative investing, it is only possible to cover a small fraction of it in a course, or even in a book. To address this problem, I made three choices.

First and most important, I aim for synthesis. A book is, first of all, a knowledge filter. In his tribute to his editor, Robert Caro, Robert writes that he wanted to title his book, *Editor*, Kelley, saying that Kelley’s gift was to make a book that was five hundred pages long, but still feels fresh and necessary today. In order to keep my book of manageable length, my working principle has been to focus on real-world problems and then use the simplest techniques that allow me to address the problem at some level of detail. This means some topics are missing in this book, or are covered at the expense of others. My choice of topics reflects my subjective view of secondary importance, material that was too hard for the payoff that it gave the reader, and also topics or ideas that are not sufficiently well-formed, or too experimental. Even if you choose not to read my book, I implore you to internalize the previous sentence. Focus on problems, not on tools. Enough said about problem selection. There are thousands and thousands of finance papers. In line with technical virtuosity but oblivious of reality. Do not fall into temptation, by applications be driven.[1]

Second, I consider risk management and portfolio management as intrinsically connected. Asset return modeling, volatility estimation, portfolio optimization, errors in data and in-sample performance analysis are all related. For example, hedging belongs to risk management, but it is also a form of portfolio construction. Therefore, portfolio construction I have avoided redundancy as much as possible. Sections often refer to earlier ones or are linked to later ones. As I was revising my book draft, whenever I found I had introduced some topic (often because I was infatuated with it) that was not connected to others, I ruthlessly cut it out. This is called “killing your darlings,” but here in reverse. That is, the subject of the book is a tree, not a pruning tool. One can grow with a bit of pruning. Out of metaphor, there is a lot of material in this book, and I am challenging at times.

Third, I occasionally integrate some standard financial results approaches with tools from the field of statistical learning. The former is applied to fundamental factor modeling, portfolio optimization, and performance attribution. I use the latter in the context of return prediction and backtesting. My hope is that the integration of these different approaches is seamless.

---
(Right Sidebar Text from Page 17)

The questions that this address in this book are:

*   How do I model returns in a way that allows me to generate risk and return forecasts?
*   What are excess returns?
*   How do I model and forecast returns?
*   How do I describe and forecast risk?
*   How do I test risk forecasts and return forecasts?
*   How do I define alpha?
*   How do I measure these signals?
*   How do I combine these signals?
*   What is the impact of risk and alpha errors on performance?
*   How do I account for transaction costs in portfolio management?
*   How do I hedge a portfolio?
*   How do I optimize?
*   How do I allocate risk over time?
*   How do I distinguish skill from luck?

The style of the book is also different. I have kept in mind the six qualities Italo Calvino proposes to apply to literature in his *Six Memos for the Next Millennium*: Lightness, Quickness, Exactitude, Visibility, Multiplicity and Consistency. My aversion to advanced mathematics notwithstanding, I must warn the reader that the book is not easy. During my lectures, I have advised more than one student not to take my course. After all, there are few financial engineering or quantitative finance positions available. Moreover, financial engineering and quantitative finance are not easily learned. A handful have become portfolio managers at hedge funds and risk managers. Yet, it is the easiest book I could write for the task, and it is written in the friendliest style I am capable of. Also, I would be lying—and contrary to the spirit of this book—if I pretended that this is a book to be read once and put aside, that the book is the last word on the subject. On the contrary, you and I are in this book together, and together we shall keep a beginner’s mind (Suzuki, 1970): a spirit of openness and curiosity, even when facing advanced topics. I will provide you with the theory behind the operations, and if you do work with me, you will gain enough skill to use the tools. If you need further help, you may remember a cyclostyled zine (Figure 1). On its second page it showed three open chords; below them, a command: “NOW FORM A BAND.” May this book be your field guide to being a pure quantitative researcher. It will be a life well lived.

---

[Image: Figure 1: Frank Zappa's Sideburn #1, page 9 (1977). Source: Dack.com/Zappa]

**Prerequisites**

The book should be accessible to a beginning graduate or advanced undergraduate student in Physics, Mathematics, Statistics, or Engineering. This means having a working relationship, and possibly a semantic one, with advanced linear algebra, probability theory, and statistics. The ideal reader has some previous interest in quantitative modeling of real-life phenomena. Many readers will be either members of a systematic trading team, or work as quantitative researchers in the central team of a hedge fund or a quantitative asset manager.

The book’s material is organized in such a way that you do not need to go through mathematical proofs. You can rely only on informal statements of mathematical results in the main body of the chapter. Some detail will be provided for some results. The reader can find the end of the chapter contains more rigorous statements, proofs, and background material. If you plan on actively doing research, you should study them, eventually.

Even if you read only the main body, you should be used to thinking in mathematical models. The Book of Nature is written in a mathematical language, Be comfortable with linear algebra, at the level of Strang (2019) and Trefethen and Bau (1997).

Some applied probability, at the level of Ross (2013). Exposure to time series, at the level of Tsay (2010), helps. Many students who come to this from economics, statistics, and finance find Hastie et al. (2017), control theory (Simoncelli), or statistics (Freydman et al., 2008).

Some asset return modeling is a plus. The first few chapters of Gned and biotechnology (2009) could be ideal. However, I will cover the basic theory in an appendix.

**Organization**

Like Caesar’s Gaul, the book is broadly divided into three parts. The first part focuses on return modeling. I cover the basics of GARCH early on because they are needed for factor modeling, and then cover factor models, both statistical and fundamental. These topics are covered in Chapters 2 through 6. Fundamental and statistical models. These topics are covered in depth, and both the treatment and some of the modeling approaches are novel. Finally, I cover data snooping/backtesting as a separate chapter, since it is a central element of the investment process.

The second part is devoted to portfolio construction and performance analysis, both ex ante and ex post. The focus is on mean-variance optimization (MVO). I emphasize the geometric interpretation of MVO, its connection to linear regression, covariance estimation, and error propagation. This allows for a synthetic, elegant characterization of performance and for concise proofs. The statistics of the Sharpe Ratio are covered in some detail. The decomposition of portfolio performance into components (factor and idiosyncratic Profit and Loss (P&L), and portfolio construction error) is also novel. As in the previous section, model error plays an important role in this part. If the optimization problem is Othello, then model error must be Iago: it can drive the optimization insane. Unlike in Shakespeare’s tragedies, we can try to rewrite the endings and turn them into comedies.

The third part is the shortest. It contains results about intertemporal volatility allocation and performance attribution. These are essential components of the investment process and belong in a book with the word “Elements” in the title.

Each chapter is organized like an article. You first read a survey the essential results in the main body of the chapter. Sections that are more advanced or sections marked with a star “*” are more advanced and can be skipped on a first reading. Proofs of new results or basic technical material are relegated to the appendices at the end of the chapters. The goal is not to disrupt the flow of learning. As mentioned at the beginning of this preface, the contents of this book are not first read in a linear fashion. I envision that the book will be used as a reference, and it should be suitable for self-study. The dependencies among the chapters are shown in Figure 2.

[Image: Figure 2: Chapter dependencies.]
(Diagram shows dependencies: 2. Basic Portfolio Management -> 3. Performance -> 4. Univariate Returns; 2 -> 5. Linear Models; 5 -> 6. Evaluating Risk; 5 -> 7. Fundamental Factor Models; 5 -> 8. Statistical Factor Models; 2 -> 10. Advanced Portfolio Management; 10 -> 11. Term Asset Allocation; 10 -> 12. Hedging; 10 -> 13. Dynamic Risk Management; 13 -> 14. Skill, Luck, Persistence)

Giuseppe ‘pino’ F. Paleologo
Raanana, New York
March 21, 2022

**Notes**

[1] References to “By Demons Be Driven” by Pantera and to Macduff’s famous speech.
[2] “Philosophy is written in this grand book, the universe, which stands continually open to our gaze. But the book cannot be understood unless one first learns to comprehend the language and read the letters in which it is composed. It is written in the language of mathematics, and its characters are triangles, circles, and other geometric figures without which it is humanly impossible to understand a single word of it; without these, one wonders about in a dark labyrinth.” (Galilei, 1623).

---

