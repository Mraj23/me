---
title: Idea Inspiration for AI Startups
---
If you're trying to figure out where to build or invest in AI for professional work, you're probably drowning in hype. Every week brings new claims about which jobs AI will automate, which industries are "ripe for disruption," and which startups are about to replace entire professions.

The reality is more nuanced—and more useful if you know where to look.

Three major benchmark studies released in late 2025 give us the clearest picture yet of what AI can and can't do in professional settings. OpenAI's **GDPval** tested AI across 44 occupations representing 75% of U.S. GDP. Mercor's **APEX** series evaluated performance in elite knowledge work—investment banking, law, consulting, and medicine. And Carnegie Mellon's **TheAgentCompany** measured how AI agents perform when dropped into realistic work environments.

Together, these benchmarks reveal two distinct ways to understand AI's current capabilities:

**The Vertical Lens:** Which industries and occupations show the strongest AI performance? This tells you where AI integration is most viable today and how to apply these models within specific professional domains.

**The Horizontal Lens:** What universal barriers limit AI across _all_ industries? This reveals the underlying technology challenges that any AI deployment—regardless of sector—must overcome.

Both lenses matter for different reasons. The vertical view helps you pick where to focus and how to design AI-assisted workflows for specific professions. The horizontal view tells you what technical barriers your product will face no matter which industry you target.

---

## The Benchmarks: What They Tested and How

Before diving into results, it's worth understanding what these studies actually measured—because the methodology shapes what the numbers mean.

### OpenAI's GDPVal 

OpenAI's GDPval evaluated **44 occupations** across **9 sectors** using **1,320 tasks** crafted by professionals averaging 14 years of experience. The study focused on sectors contributing over 5% to U.S. GDP, covering occupations that collectively earn $3 trillion annually.

**How tasks were created:** Industry experts designed tasks based on their actual work product. Each task went through multiple rounds of review—an average of five human reviews per task—to ensure realism and quality. Tasks required an average of 7 hours of professional work to complete, with some spanning multiple weeks. Experts classified their tasks against O*NET occupational categories to ensure broad coverage of typical work activities.

**What a task looks like:** Tasks ranged across professions and included things like:

- A **Financial Analyst** creating a competitor landscape analysis for last-mile delivery companies
- A **Registered Nurse** assessing skin lesion images and creating a consultation report
- A **Real Estate Agent** designing a sales brochure for a new DC property
- A **Customer Service Representative** drafting an email response to a dissatisfied customer requesting a return

Each task came with reference files—up to 17 documents in the gold subset—that models needed to parse and use. These included spreadsheets, PDFs, images, slide decks, and other formats typical of professional work.

**How grading worked:** GDPval used **blind pairwise expert comparisons**. An industry expert would receive the original task request, the reference files, and two unlabeled work deliverables—one produced by a human professional, one by an AI model. Without knowing which was which, the expert ranked which deliverable was better (or marked them as equivalent). On average, each comparison took over an hour to grade.

The key metric is **Human Parity Percentage**—the win/tie rate indicating how often expert graders preferred the AI's output or judged it equivalent to human work.

**Critical point: 50% represents human parity.** A score of 50% means graders couldn't tell the difference between AI and human output. Above 50% means AI won more often than humans. Below 50% means humans consistently outperformed AI.

### Mercor's APEX Benchmark

Mercor's APEX focused on four high-value knowledge professions: **investment banking, law, management consulting, and medicine**. The benchmark used 400 tasks (100 per profession) requiring 1-8 hours of professional time to complete.

**How tasks were created:** 256 domain experts—including former consultants from BCG and McKinsey, investment bankers from Morgan Stanley and Citigroup, and corporate lawyers at Fortune 500 companies—designed tasks based on their real-world work. Average expert experience was 12.9 years. Each task came with source documents (PDFs, spreadsheets, financial models) averaging 26,677 tokens of context.

**What a task looks like:** Here's an actual example from the investment banking category:

> **Prompt:** "TPG is currently in conversations to acquire a wholesale distributor of HVAC supplies in Europe. Management has shared company projections, which the investment team has used to build out an LBO model. Before making a decision, TPG wants to model out a few scenarios to understand the return profile and carry implications for them..."
> 
> The task then specifies two exit scenarios (FY28 and FY29) and asks the model to calculate Net IRR for LPs, Net MOIC for LPs, Carry for the GP, and percentage of value created due to EBITDA growth vs. Multiple Expansion.

**How grading worked:** Unlike GDPval's head-to-head comparison, APEX used **rubric-based scoring**. Each task had a detailed grading rubric with objective pass/fail criteria—averaging 29 criteria per task in the original APEX-v1-extended and about 4 criteria per task in APEX-Agents. For the investment banking example above, criteria included specific acceptable ranges for each calculation (e.g., "Net IRR for LPs is 29.79%, acceptable range is 29.49% to 30.09%").

Each model ran each task 8 times, with scores averaged to account for variability. An LLM judge (Gemini 2.5 Pro) evaluated outputs against the rubrics.

APEX scores represent the **mean percentage of criteria met**—so 67% means the model satisfied roughly two-thirds of the objective quality checks on average.

**Two evaluation modes:** APEX ran two versions that reveal something crucial:

- **APEX-v1-extended** (single-turn): Models received complete prompts with all necessary information and source files attached directly
- **APEX-Agents** (multi-step): Models had to navigate realistic work environments with 166 files across 9 applications, finding information themselves

The gap between these modes tells us what happens when AI moves from ideal conditions to realistic deployment.

### CMU's TheAgentCompany

Carnegie Mellon's TheAgentCompany (I was one of the authors on this paper) dropped AI agents into a simulated software company with real applications—GitLab for code management, Plane for project tracking, RocketChat for team communication, and ownCloud for document storage. The benchmark included 175 tasks across software engineering, project management, HR, finance, and administrative work.

**How it worked:** Unlike the other benchmarks where models received pre-packaged information, agents were given a goal that required navigating real software, browse the web, write code, run programs, and communicate with simulated coworkers to complete tasks.

**Evaluation:** Tasks used checkpoint-based scoring—agents earned partial credit for completing intermediate steps toward the final goal. Two metrics matter:

- **Full completion score:** Did the agent pass all checkpoints?
- **Partial completion score:** How much progress did the agent make?

This benchmark reveals the gap between "AI can do this task" and "AI can do this task in a realistic work environment."

---

## Part 1: The Vertical Lens—Where AI Performs Best

### GDPval Results by Occupation

GDPval's results show substantial variation across occupations. Here's where AI performed strongest:

|Occupation|Human Parity %|Total Compensation|Best Model|
|---|---|---|---|
|Counter and Rental Clerks|**81%**|$17.42B|Claude|
|Sales Managers|**79%**|$97.16B|GPT-5|
|Shipping, Receiving, and Inventory Clerks|**76%**|$38.50B|Claude|
|Editors|**75%**|$8.18B|GPT-5|
|Software Developers|**70%**|$239.18B|Tie|
|Private Detectives and Investigators|**70%**|$2.39B|Claude|
|Compliance Officers|**69%**|$33.80B|GPT-5|
|First-Line Supervisors of Non-Retail Sales|**69%**|$21.43B|Claude|
|Sales Reps, Wholesale (Except Tech)|**68%**|$103.21B|GPT-5|
|General and Operations Managers|**67%**|$477.16B|Claude|

And where AI struggled most:

|Occupation|Human Parity %|Total Compensation|Best Model|
|---|---|---|---|
|Industrial Engineers|**17%**|$37.79B|Claude|
|Film and Video Editors|**17%**|$2.41B|Tie|
|Accountants and Auditors|**24%**|$135.44B|Claude|
|Mechanical Engineers|**25%**|$31.57B|GPT-5|
|Pharmacists|**26%**|$45.12B|Claude|
|Order Clerks|**28%**|$3.86B|Claude|
|Concierges|**29%**|$1.80B|Claude|

### Weighting GDPval Results by Economic Scale

Raw performance percentages don't tell the whole story. A job where AI performs at 70% but only employs a handful of people matters less than one at 50% that represents hundreds of billions in compensation.

The **AI Market** metric (Total Compensation × Human Parity Percentage) reveals where capability meets scale:

|Rank|Occupation|AI Market|Human Parity %|
|---|---|---|---|
|1|General and Operations Managers|**$319.70B**|67%|
|2|Software Developers|**$167.43B**|70%|
|3|Registered Nurses|**$119.53B**|37%|
|4|Sales Managers|**$76.76B**|79%|
|5|Customer Service Representatives|**$72.98B**|59%|
|6|Sales Reps, Wholesale (Except Tech)|**$70.18B**|68%|
|7|Computer and Information Systems Managers|**$63.15B**|52%|
|8|Lawyers|**$62.86B**|46%|
|9|Medical and Health Services Managers|**$50.65B**|65%|
|10|Financial Managers|**$47.28B**|32%|

Some surprises here. Registered Nurses rank #3 by AI Market despite only 37% parity—the sheer scale of nursing ($323B in total compensation) makes even modest AI assistance economically significant. Conversely, Editors at 75% parity barely crack the top 40 because the total market is small ($8.18B).

This AI Market metric is admittedly naive—it doesn't account for task complexity, adoption barriers, or how much of each profession's work is actually automatable. But it provides a rough sense of where AI capability intersects with economic scale, which matters for prioritizing where to build.

### APEX Results: Elite Professional Work

In the elite professions APEX tested, performance varied by domain:

|Occupation|Best Model|Score (% Criteria Met)|
|---|---|---|
|Law|GPT 5 (Thinking = High)|77.9%|
|Medicine|GPT 5 (Thinking = High)|65.5%|
|Management Consulting|Gemini 3 Pro (Thinking = High)|64.0%|
|Investment Banking|Gemini 3 Pro (Thinking = High)|63.0%|

---

## What the Vertical Results Actually Mean for Builders

### 1. These Models Work as Copilots, Not Replacements

There's a crucial detail about how both GDPval and APEX ran their experiments that changes how you should interpret the results: **every task came with crystal-clear instructions.**

As the GDPval paper acknowledges in its limitations section: "Tasks are precisely-specified and one-shot, not interactive: For GDPval, we provide the full context of the task in the prompt, but in real life it often takes effort to figure out the full context of a task and understand what to work on."

In the real world, a lawyer doesn't get a neat prompt with all the documents pre-selected and a rubric attached—they get a stressed-out client with a messy story and have to figure out what legal questions even matter. A nurse doesn't get a structured task description—they observe a patient and have to decide what's actually wrong. The benchmark tested whether AI can _do_ the work when everything is spelled out. They didn't test whether AI can _figure out_ what work needs doing.

GDPval actually ran a separate experiment to confirm this gap. They created "under-contextualized" versions of their prompts—"shorter prompts \[that] omitted additional context such as where to locate specific data within reference files, how to approach the problem, or detailed formatting expectations for the final deliverable; the models had to 'figure it out.'" On average, these revised prompts were 42% the length of the originals. The result? "The model's performance was worse on under-specified prompts. In particular, the models struggled to figure out context."

**What this means for builders:** The opportunity here looks more like traditional SaaS software than autonomous AI agents. These models can't automate a job end-to-end, but they can dramatically accelerate specific tasks when a human stays in the loop to provide context and verify outputs. That means pricing and go-to-market should follow SaaS patterns—you're selling efficiency gains and productivity tools, not labor replacement. The value proposition is helping a professional do their job faster, not eliminating the professional entirely.

### 2. The Professional Knowledge is Already Baked In—Your Moat is the Custom Data

Here's something that should change how you think about building in this space: frontier models already know how to do these jobs.

Look at the scores again. GPT-5 hit 77.9% on legal tasks. Claude reached 81% on real estate workflows. Software development came in at 70%. These models weren't fine-tuned on law firm documents or trained specifically on property management—they absorbed professional knowledge from their general training data. They already understand what a legal brief looks like, how financial models work, and what good code structure means.

This is both good and bad news for builders:

**The good news:** You don't need to teach AI the fundamentals of your target profession. The foundation is there. A legal AI startup doesn't need to start from "what is a contract?"—GPT-5 already knows.

**The bad news:** Everyone else has access to that same foundation. If your entire value proposition is "AI that understands legal work," you're competing with every other company using the same base models.

**Where the real opportunity lies:** The generic models perform well on generic tasks. But real businesses have specific ways of doing things—particular document templates, communication styles, client preferences, regulatory quirks. A model that scores 70% on generic consulting tasks might drop to 50% when faced with a specific firm's proprietary frameworks.

The winning strategy is building a data moat on top of the foundation:

- Fine-tune on a law firm's historical briefs, not just general legal documents
- Train on a hospital's actual care plan templates, not textbook examples
- Capture the specific preferences of individual partners, doctors, or managers through feedback loops

The foundation is commodity. The moat is in the specifics. You're not teaching AI "what is legal analysis"—you're teaching it "how this particular partner likes arguments structured, what tone this client expects, and what formatting quirks this jurisdiction requires."

### 3. Reliability Can Be Improved Through Applied AI Techniques

There's a tempting but wrong conclusion you might draw from these results: that AI is "ready" for low-stakes work at 60-70% accuracy but needs 99%+ for high-stakes industries like medicine or law.

This misses the point. **No business wants to ship work that fails 30% of the time, regardless of industry.**

Think about it practically. A 30% error rate on lease documents means angry tenants, legal liability, and constant rework. A 30% error rate on sales presentations means lost deals and embarrassed sales teams. Yes, a 30% error rate in medical diagnosis has more severe individual consequences—but that doesn't mean property managers are happy with unreliable output. 

The good news is that applied AI techniques—prompting, scaffolding, and fine-tuning—can meaningfully close this gap. GDPval found that relatively simple interventions made a real difference: "Prompting fully eliminated black-square artifacts from GPT-5 responses, which previously affected over half of generated PDFs, and reduced egregious formatting errors in PowerPoint files from 86% to 64%... Prompting also improved human preference win rates by 5 percentage points." The researchers concluded that "these easy performance gains suggest there are paths to agent improvement on GDPval tasks by training or scaffolding them to be more thorough and take full advantage of their multimodal capabilities."

This means there's real value in the engineering work of optimizing AI for specific use cases. Off-the-shelf models get you to 60-70% parity; careful prompting, domain-specific fine-tuning, and thoughtful scaffolding can push that higher. The companies that invest in this applied work will outperform those that just wrap a generic API.

The deployment architecture remains human-in-the-loop regardless:

1. **Human defines the task** and provides the structured context
2. **AI generates a draft**
3. **Human reviews and approves**

But the better your applied AI work, the less time humans spend fixing errors in step 3—and that's where the efficiency gains compound.

### The Vertical Opportunity

The occupations with the highest AI Market values—General and Operations Managers ($319B), Software Developers ($167B), Registered Nurses ($119B), Sales Managers ($76B)—represent the most attractive targets for AI copilot tools. These combine meaningful AI capability with massive economic scale.

The opportunity is to work closely with enterprises in these high-value segments to deploy solutions tailored to their specific needs. Generic AI tools will capture some value, but the real wins come from understanding a particular company's workflows, document standards, and quality thresholds—then building a solution that hits an acceptable parity level for their business. That might mean 70% parity is fine for internal draft generation but 90%+ is required for client-facing deliverables. The companies that can navigate these requirements and deliver customized solutions will capture disproportionate value.

---

## Part 2: The Horizontal Lens—Universal Barriers to AI Agents

The vertical lens shows which occupations and industries are best suited for AI assistants today. The horizontal lens reveals the general challenges across all industries that must be solved to transition AI from simple assistants or copilots to autonomous agents capable of joining the labor force.

Four universal barriers emerge across all three benchmarks—and they affect every industry equally.

### Barrier 1: The Environment Gap—From Assistants to Agents

The most dramatic finding comes from comparing APEX's two evaluation modes on identical professional categories. The difference in task structure tells the story:

**APEX-v1-extended Task Structure (Assistant Mode):**

```
USER: [Detailed prompt with all necessary information]
ATTACHMENTS: [5-10 source documents provided]
EXPECTED: [Written analysis, completed document]
MODEL: [Reads, reasons, writes answer]
RESULT: 67% average success
```

**APEX-Agents Task Structure (Agent Mode):**

```
USER: [Brief task description]
ENVIRONMENT: [Navigate workspace averaging 166 files across 9 applications]
EXPECTED: [Completed deliverable after multi-step workflow]
MODEL: [Must find files, coordinate tools, maintain context across 20-50 steps]
RESULT: 24% average success
```

The performance breakdown by profession:

|Job Category|Single-Turn (Clear Prompts)|Multi-Step (Real Environment)|Performance Gap|
|---|---|---|---|
|Investment Banking|63.0%|27.3%|**−35.7 pp**|
|Management Consulting|64.0%|22.7%|**−41.3 pp**|
|Corporate Law|77.9%|25.9%|**−52.0 pp**|

That's an average drop of **43 percentage points** when agents have to navigate real work environments instead of receiving pre-packaged information.

TheAgentCompany's benchmark provides additional granularity on where environment navigation fails. The table below shows success rates of the best-performing models on tasks that required using specific software applications within the simulated company:

|Software Environment|Success Rate (Best Model)|
|---|---|
|Plane (Project Tracking)|41.2%|
|GitLab (Code Management)|33.8%|
|RocketChat (Communication)|29.1%|
|ownCloud (Office Suite)|12.9%|

The pattern here is significant. Plane and GitLab are developer-centric tools—software engineers have been building AI coding assistants and automation for these environments for years. The interfaces are relatively structured, the APIs are well-documented, and there's substantial training data available. Agents have learned to navigate these environments reasonably well.

ownCloud, by contrast, is an office suite similar to Google Drive or Microsoft 365—the kind of environment where most professional work actually happens. Documents, spreadsheets, presentations, shared folders, version histories. These interfaces weren't designed with AI agents in mind. They're built for humans clicking through menus and dragging files between folders. TheAgentCompany researchers noted that even simple UI elements become obstacles: "On many tasks that involve ownCloud, a closable welcome popup has become an obstacle for [the] agent which uses text-based browsing. [The] agent gets stuck and fails to click on the 'x' to close the popup." The 12.9% success rate reflects how poorly current agents handle these unstructured, visually-oriented environments.

This matters because most knowledge work doesn't happen in GitLab. It happens in Google Workspace, Microsoft 365, Salesforce, and Workday. The benchmarks showing 70%+ performance measure what AI can do when a human has already gathered the context and handed it over. The 12.9% ownCloud figure is closer to what real-world agent deployment looks like today.

### Barrier 2: Tool Use Failures and Consistency Gaps

Successful tool use is far from a solved problem. Even when agents can use individual tools, they struggle to combine them effectively. APEX-Agents found a counterintuitive pattern:

|Metric|Successful vs. Failed Attempts|
|---|---|
|Unique tools used|+0.22 more|
|Total tool calls|**−5.66 fewer**|
|Total steps|**−5.95 fewer**|

Successful task completion correlates with using _fewer_ total tools and steps, not more. Failed attempts are characterized by what the researchers call "doom looping"—repeatedly calling the same unproductive tools until hitting timeout limits.

A typical failed pattern:

1. Query a file for information (not found)
2. Query the same file again (still not found)
3. Query again (still not found)
4. Continue until 250-step timeout

A human would recognize after the first attempt that the file doesn't have what they need and try somewhere else. Current AI agents often can't make that adaptive shift.

The consistency gap compounds the problem:

- **Pass@8** (succeeds at least once in 8 attempts): ~40%
- **Pass^8** (succeeds all 8 times): ~13%

That 27-percentage-point gap means agents can sometimes complete tasks but can't do so reliably. The same agent given the same task will succeed on some attempts and fail on others with no predictable pattern. For any production deployment, this inconsistency is a dealbreaker—you can't build workflows around tools that work randomly.

### Barrier 3: Instruction Following—The Most Common Failure Mode

Across all benchmarks, instruction-following failures beat out accuracy errors as the primary reason AI outputs got rejected:

|Model|Instruction Following Failures|Formatting Failures|Accuracy Failures|
|---|---|---|---|
|Claude Opus 4.1|14%|5-6%|~5%|
|GPT-5 high|9-10%|5%|~5%|
|Grok 4|35%|13%|~5%|
|Gemini 2.5 Pro|40%|6-7%|~7%|

Models don't primarily fail because they get facts wrong. They fail because they don't follow explicit instructions—ignoring reference data, using the wrong format, or promising deliverables they never provide.

TheAgentCompany documented an even more concerning pattern: when agents get stuck, they sometimes try to "cheat" by creating shortcuts that bypass the actual task:

> *"The agent cannot find the right person to ask questions on RocketChat. As a result, it then decides to create a shortcut solution by renaming another user to the name of the intended user."*

Rather than acknowledging they're stuck, agents may take actions that superficially satisfy completion criteria while completely missing the actual objective. This kind of failure is harder to catch than a simple error—the output looks correct until you realize the agent gamed its way to a "solution."

### Barrier 4: Time Horizon—Performance Decays with Task Length

GDPval's data shows consistent performance degradation as tasks get longer:

|Task Duration|Claude Opus 4.1 Win Rate|Drop from Baseline|
|---|---|---|
|0-2 hours|56%|—|
|2-4 hours|44%|−12 pp|
|4-8 hours|42%|−14 pp|
|8+ hours|37%|−19 pp|
Task duration predicts AI performance better than industry or occupation category. Most meaningful professional work takes more than 2 hours—the tasks where AI performs best in these benchmarks represent a subset of actual professional work. As task complexity and duration increase, the barriers compound: longer tasks require more workspace navigation, more tool orchestration, and more sustained instruction-following.

### The Horizontal Opportunity

These four barriers—environment navigation, tool use failures, instruction following, and time horizon decay—are not specific to any industry. 

For researchers and builders exploring the agent space, this universality is the opportunity. Agent frameworks, harnesses, or orchestration systems that solve these problems can win across all sectors. A breakthrough in reliable tool use doesn't just help legal AI—it helps every vertical. Better approaches to long-horizon task planning apply to consulting agents and healthcare agents alike.

What's interesting is that these challenges look more like fundamental research problems than capability limitations of the underlying language models. The models themselves reason well—67% success on single-turn professional tasks demonstrates that. The failures emerge in the harnesses, orchestration systems, tool interfaces, and planning architectures that sit around the models. This means the agent problem isn't exclusively the domain of large labs training ever-bigger models. Researchers and builders outside the major AI companies can meaningfully contribute by designing better agent frameworks, more robust tool-use patterns, and architectures that maintain instruction fidelity across extended workflows.

---

## Conclusion: Two Paths Forward

The benchmarks reveal two distinct opportunities with different timelines and requirements.

**The near-term path** is building copilots and assistants. AI can already complete professional tasks at or near human parity—but only when a human provides clear instructions and reviews the output. This means SaaS-level pricing, not labor replacement. The value is augmenting workers, not automating them. And building a sticky product requires deep vertical integration: understanding specific industry workflows, fine-tuning on company-specific documents, and delivering outputs tailored to each customer's quality standards.

**The longer-term path** is solving the agent problem. To have AI that truly substitutes for labor, the horizontal barriers must be overcome. An agent that succeeds 24% of the time on realistic work tasks cannot reliably replace a knowledge worker—the economics don't work. But because these barriers are universal across industries, the companies and research groups that solve them will unlock value across every vertical simultaneously.

For now, the practical approach is building copilots that work within human-in-the-loop constraints while the harder agent problems get solved. The builders who understand both lenses—where AI capability is strong today, and what barriers remain before autonomy becomes viable—are the ones positioned to capture value in both the near and long term.