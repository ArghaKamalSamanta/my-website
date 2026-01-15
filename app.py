from flask import Flask, render_template, request, redirect, url_for, flash
import os

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "replace_this_with_env_secret")

# ---------- Data pulled from resume (editable) ----------
PERSON = {
    "name": "Argha Kamal Samanta",
    "title": "AI / ML Researcher • Final year Student at IIT Kharagpur",
    "email": "arghakamal25@gmail.com",
    "phone": "+91 xxx-xxx-xxxx",
    "linkedin": "https://linkedin.com/in/argha-kamal-samanta-38644b237/",
    "github": "https://github.com/ArghaKamalSamanta",
    "google_scholar": "https://scholar.google.com/citations?user=0kmgF3UAAAAJ&hl=en",
    "location": "IIT Kharagpur (Dept. of E & EC)",
    "summary": "B.Tech in Electronics & Electrical Communication Engineering + M.Tech in Vision & Intelligent Systems. AI / ML engineer with internship & research experience in LLM systems, on-device inference, and applied ML in vision, health & finance.",
    # Quick lists for template rendering
    "skills": [
        "Python", "C", "C++", "Matlab", "PyTorch", "TensorFlow", "HuggingFace", "LangChain", "Pandas",
        "Scikit-learn", "OpenCV", "Flask", "Matplotlib", "XGBoost", "ONNX"
    ],
    "education": [
        {"degree": "B.Tech + M.Tech", "institute": "Indian Institute of Technology Kharagpur", "grade": "8.62/10", "year": "2026"},
        {"degree": "XII (WBCHSE)", "institute": "Ramakrishna Mission Vidyapith Purulia", "grade": "96.20%", "year": "2021"},
        {"degree": "X (WBBSE)", "institute": "Ramakrishna Mission Vidyapith Purulia", "grade": "95.57%", "year": "2019"},
    ],
    "experience": [
        {
            "role": "Student Trainee — Samsung Research Institute Bangalore",
            "period": "May 2025 – Jul 2025",
            "bullets": [
                "Built a PoC for on-device LLM inference on memory-constrained devices via dynamic model subparts loading.",
                "Split LLaMA-3.2-3B into ONNX subgraphs (embedding, grouped decoders, output) to enable controlled memory usage.",
                "Implemented asynchronous subpart loading- unloading using multithreading and ONNX Runtime.",
                "Achieved ∼ 50% memory usage reduction with only ∼ 6% increase in inference time compared to full model load."
            ]
        },
        {
            "role": "Undergraduate Researcher — AI Institute, University of South Carolina",
            "period": "May 2024 – Feb 2025",
            "bullets": [
                "Conducted research on the capabilities of Large Language Models to retain proper context alignment in its response.",
                "Proposed a novel metric to quantify LLMs’ context-picking ability utilizing KL Divergence b/w context and response.",
                "Utilized Direct Policy Optimization to train LLMs like Mistral, Llama, Qwen, Gemma for better context alignment."
            ]
        },
        {
            "role": " AI Engineer Intern — Aqxle AI",
            "period": "Dec 2024 – Dec 2024",
            "bullets": [
                "Built an LLM agent that enables querying CSV files using natural language to generate consistent, code-driven insights.",
                "Designed system prompt and utility codes to enforce consistent outputs, and plots and tabular responses when appropriate.",
                "Integrated GPT-4 Turbo with pandas preprocessing to dynamically handle date parsing, plotting, and tabular summaries.",
                "Designed backend logic for chat history tracking, code segmentation, file storage, and plot generation for user queries."
            ]
        },
        {
            "role": " AI Engineer Intern — Ema Unlimited",
            "period": "Aug. 2024 – Dec. 2024",
            "bullets": [
                "Developed an LLM Agent to automate PR code reviews, optimization suggestions, and changelog updates.",
                "Engineered a GitHub app for automated PR analysis, issue detection, and performance feedback using single-call LLMs.",
                "Built a GitHub tool to auto-generate PR descriptions, reviews, and suggestions, boosting efficiency and code quality."
            ]
        },
        {
            "role": " Machine Learning Intern — Stanford University School of Medicine",
            "period": "May 2023 – Aug. 2023",
            "bullets": [
                "Hypothesized and monitored maternal and child health in low-income countries using satellite and geotagged data sources.",
                "Employed feature engineering, null-value imputation and hyperparameter tuning on XGBoost and LightGBM.",
                "Achieved an RMSE of 11.889 with XGBoost on big data analysis containing 120k examples and 11k+ features.",
                "Successfully predicted health indicators like mean BMI, median BMI, under-five mortality rate, skilled birth-attendant rate."
            ]
        },
    ],
    "projects": [
        {
            "name": "Optimal Trade Execution & Market Impact Modeling",
            "meta": "Blockhouse Capital — Aug. 2025",
            "desc": "At Blockhouse Capital, I designed and implemented an intraday optimal trade execution framework integrating statistical learning, dynamic optimization, and adaptive control. Temporary market impact functions were estimated using XGBoost, symbolic regression, and Explainable Boosting Regression (EBR), with EBR achieving an R^2 = 0.82 and yielding interpretable piecewise-linear approximations of the impact surface. These models were embedded into a stochastic control pipeline combining dynamic programming (DP) and model predictive control (MPC) to solve risk-adjusted execution problems under uncertainty, minimizing expected slippage relative to benchmark trajectories. To ensure robustness in non-stationary environments, the framework incorporated convex relaxations of non-linear constraints and evolutionary search heuristics, augmented with online learning updates for continual parameter adaptation to real-time market dynamics.",
            "link": "https://github.com/ArghaKamalSamanta/blockhouse_qn1.git"
        },
        {
            "name": "Stock Return Forecasting",
            "meta": "Trexquant Investment LP Kaggle Challenge — Jul. 2024",
            "desc": "At Trexquant Investment LP Kaggle Challenge, I developed a machine learning pipeline for stock return forecasting on earnings announcement events. I conducted extensive exploratory data analysis to identify weak predictive structure in the dataset and engineered temporal features including lagged returns, rolling means, and exponential moving averages (EMA) to attenuate noise and capture short-term dynamics. The predictive models were built using ensemble methods, primarily XGBoost and Gradient Boosting, which demonstrated robustness to the low signal-to-noise ratio and nonlinear feature interactions. Despite limited feature–target correlation, the framework achieved an RMSE of 0.0041, highlighting the efficacy of feature smoothing and boosting ensembles in financial return prediction under noisy regimes.",
            "link": "https://github.com/ArghaKamalSamanta/Stock-Return-Forecasting-During-Earnings-Announcements.git"
        },
        {
            "name": "StockInsight Pro: LLM-driven Fundamental Analysis",
            "meta": "FSIL, Georgia Institute of Technology — Apr. 2024",
            "desc": "At FSIL, Georgia Institute of Technology, I built StockInsight Pro, a large language model (LLM)-driven framework for fundamental equity analysis. The system integrated the SEC EDGAR API to retrieve 10-K filings spanning 1995–2023, thereby establishing a robust historical corpus for financial research. To enable efficient document interaction, I constructed a vector database using Chroma and applied retrieval-augmented generation (RAG) techniques to ground LLM outputs in primary source filings. The analytical core employed mistral-7b-instruct-v0.1, an open-source instruction-tuned model, to extract and interpret financial insights at scale. The platform was designed not only to provide interactive querying of regulatory filings but also to generate buy–sell recommendations informed by automated fundamental analysis, bridging traditional financial statement analysis with modern generative AI capabilities.",
            "link": "https://github.com/ArghaKamalSamanta/GeorgiaTech_task.git"
        },
        {
            "name": "Contrail Detection in Infrared Satellite Images",
            "meta": "Google Research Kaggle Challenge — May 2023",
            "desc": "At the Google Research Kaggle Challenge, I developed a hybrid deep learning framework for contrail detection in infrared satellite imagery. The architecture combined ResNet-based convolutional layers for spatial feature extraction with Temporal Convolutional Networks (TCNs) to capture sequential dependencies across time, enabling the model to detect subtle temporal variations indicative of contrail formation. The system was trained on a dataset of over 20,000 sequences, each consisting of nine images sampled at 10-minute intervals, thereby incorporating both spatial and temporal structure into the learning process. By integrating TCN modules with CNN feature maps, the framework achieved a 4.1% improvement in detection accuracy relative to baseline CNN models, demonstrating the efficacy of hybrid spatiotemporal architectures for satellite-based atmospheric event recognition.",
            "link": None
        }
    ],
    "publications": [
        {
            "title": "Wave2Word: A Multimodal Transformer Framework for Joint EEG-Text Alignment and Multi-Task Representation Learning in Neurocritical Care",
            "meta": "arXiv 2026",
            "abstract": "Continuous electroencephalography (EEG) is routinely used in neurocritical care to monitor seizures and other harmful brain activity, including rhythmic and periodic patterns that are clinically significant. Although deep learning methods have achieved high accuracy in seizure detection, most existing approaches remain seizure-centric, rely on discrete-label supervision, and are primarily evaluated using accuracy-based metrics. A central limitation of current EEG modeling practice is the weak correspondence between learned representations and how EEG findings are interpreted and summarized in clinical workflows. Harmful EEG activity exhibits overlapping patterns, graded expert agreement, and temporal persistence, which are not well captured by classification objectives alone. This work proposes a multimodal EEG representation learning framework that integrates signal-domain modeling with structured clinical language supervision. First, raw EEG is transformed into a longitudinal bipolar montage and time-frequency representations. Second, dual transformer-based encoders model complementary temporal and frequency-centric dependencies and are fused using an adaptive gating mechanism. Third, EEG embeddings are aligned with structured expert consensus descriptions through a contrastive objective. Finally, an EEG-conditioned text reconstruction loss is introduced as a representation-level constraint alongside standard classification loss. Experimental evaluation using a controlled train-validation-test split achieves a six-class test accuracy of 0.9797. Ablation analyses show that removing contrastive alignment reduces cross-modal retrieval performance from Recall@10 of 0.3390 to 0.0045, despite minimal change in classification accuracy. These findings demonstrate that discriminative accuracy does not reliably reflect representation quality for clinically meaningful EEG modeling.",
            "authors": "Argha Kamal Samanta, Deepak Mewada, Monalisa Sarma, Debasis Samanta",
            "link": "https://arxiv.org/abs/2601.00670"
        },
        {
            "title": "Clinically Calibrated Machine Learning Benchmarks for Large-Scale Multi-Disorder EEG Classification",
            "meta": "arXiv 2025",
            "abstract": "Clinical electroencephalography is routinely used to evaluate patients with diverse and often overlapping neurological conditions, yet interpretation remains manual, time-intensive, and variable across experts. While automated EEG analysis has been widely studied, most existing methods target isolated diagnostic problems, particularly seizure detection, and provide limited support for multi-disorder clinical screening.\nThis study examines automated EEG-based classification across eleven clinically relevant neurological disorder categories, encompassing acute time-critical conditions, chronic neurocognitive and developmental disorders, and disorders with indirect or weak electrophysiological signatures. EEG recordings are processed using a standard longitudinal bipolar montage and represented through a multi-domain feature set capturing temporal statistics, spectral structure, signal complexity, and inter-channel relationships. Disorder-aware machine learning models are trained under severe class imbalance, with decision thresholds explicitly calibrated to prioritize diagnostic sensitivity.\nEvaluation on a large, heterogeneous clinical EEG dataset demonstrates that sensitivity-oriented modeling achieves recall exceeding 80% for the majority of disorder categories, with several low-prevalence conditions showing absolute recall gains of 15-30% after threshold calibration compared to default operating points. Feature importance analysis reveals physiologically plausible patterns consistent with established clinical EEG markers.\nThese results establish realistic performance baselines for multi-disorder EEG classification and provide quantitative evidence that sensitivity-prioritized automated analysis can support scalable EEG screening and triage in real-world clinical settings.",
            "authors": "Argha Kamal Samanta, Deepak Mewada, Monalisa Sarma, Debasis Samanta",
            "link": "https://arxiv.org/abs/2512.22656"
        },
        {
            "title": "Beyond CLIP: Knowledge-Enhanced Multimodal Transformers for Cross-Modal Alignment in Diabetic Retinopathy Diagnosis",
            "meta": "arXiv 2025",
            "abstract": "Diabetic retinopathy (DR) is a leading cause of preventable blindness worldwide, demanding accurate automated diagnostic systems. While general-domain vision-language models like Contrastive Language-Image Pre-Training (CLIP) perform well on natural image tasks, they struggle in medical domain applications, particularly in cross-modal retrieval for ophthalmological images. We propose a novel knowledge-enhanced joint embedding framework that integrates retinal fundus images, clinical text, and structured patient data through a multimodal transformer architecture to address the critical gap in medical image-text alignment. Our approach employs separate encoders for each modality: a Vision Transformer (ViT-B/16) for retinal images, Bio-ClinicalBERT for clinical narratives, and a multilayer perceptron for structured demographic and clinical features. These modalities are fused through a joint transformer with modality-specific embeddings, trained using multiple objectives including contrastive losses between modality pairs, reconstruction losses for images and text, and classification losses for DR severity grading according to ICDR and SDRG schemes. Experimental results on the Brazilian Multilabel Ophthalmological Dataset (BRSET) demonstrate significant improvements over baseline models. Our framework achieves near-perfect text-to-image retrieval performance with Recall@1 of 99.94% compared to fine-tuned CLIP's 1.29%, while maintaining state-of-the-art classification accuracy of 97.05% for SDRG and 97.97% for ICDR. Furthermore, zero-shot evaluation on the unseen DeepEyeNet dataset validates strong generalizability with 93.95% Recall@1 versus 0.22% for fine-tuned CLIP. These results demonstrate that our multimodal training approach effectively captures cross-modal relationships in the medical domain, establishing both superior retrieval capabilities and robust diagnostic performance.",
            "authors": "Argha Kamal Samanta, Harshika Goyal, Vasudha Joshi, Tushar Mungle, Pabitra Mitra",
            "link": "https://arxiv.org/abs/2512.19663"
        },
        {
            "title": "RADIANT: Retrieval AugmenteD entIty-context AligNmenT - Introducing RAG-ability and Entity-Context Divergence",
            "meta": "arXiv 2025",
            "abstract": "As Large Language Models (LLMs) continue to advance, Retrieval-Augmented Generation (RAG) has emerged as a vital technique to enhance factual accuracy by integrating external knowledge into the generation process. However, LLMs often fail to faithfully integrate retrieved evidence into their generated responses, leading to factual inconsistencies. To quantify this gap, we introduce Entity-Context Divergence (ECD), a metric that measures the extent to which retrieved information is accurately reflected in model outputs. We systematically evaluate contemporary LLMs on their ability to preserve factual consistency in retrieval-augmented settings, a capability we define as RAG-ability. Our empirical analysis reveals that RAG-ability remains low across most LLMs, highlighting significant challenges in entity retention and context fidelity. This paper introduces Radiant (Retrieval AugmenteD entIty-context AligNmenT), a novel framework that merges RAG with alignment designed to optimize the interplay between retrieved evidence and generated content. Radiant extends Direct Preference Optimization (DPO) to teach LLMs how to integrate provided additional information into subsequent generations. As a behavior correction mechanism, Radiant boosts RAG performance across varied retrieval scenarios, such as noisy web contexts, knowledge conflicts, and hallucination reduction. This enables more reliable, contextually grounded, and factually coherent content generation.",
            "authors": "Vipula Rawte, Rajarshi Roy, ..., Argha Kamal Samanta, ...",
            "link": "https://arxiv.org/abs/2507.02949"
        }
    ]
}
# ------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", person=PERSON)

@app.route("/about")
def about():
    return render_template("about.html", person=PERSON)

@app.route("/projects")
def projects():
    return render_template("projects.html", person=PERSON)

@app.route("/publications")
def publications():
    return render_template("publications.html", person=PERSON)

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name", "")
        email = request.form.get("email", "")
        message = request.form.get("message", "")
        # For now, we just flash a message. In prod, forward to email or store in DB.
        flash("Thanks! Your message has been received. I'll get back to you soon.", "success")
        # Optionally: store in DB, send via SMTP or third-party (SendGrid)
        return redirect(url_for("contact"))
    return render_template("contact.html", person=PERSON)

# Basic health check route (useful for Render)
@app.route("/health")
def health():
    return "OK", 200

if __name__ == "__main__":
    # For local dev only; Render will use gunicorn
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)


