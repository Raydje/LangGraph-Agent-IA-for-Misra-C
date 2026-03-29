SEED_RULES = [
    {
        "rule_id": "DO178B-REQ-001",
        "standard": "DO-178B",
        "section": "6.3.1",
        "dal_level": "A",
        "title": "Requirements-Based Testing - MC/DC Coverage",
        "full_text": (
            "For Level A software, structural coverage analysis shall demonstrate "
            "Modified Condition/Decision Coverage (MC/DC). Every condition in a "
            "decision must be shown to independently affect the outcome of the "
            "decision. This is in addition to decision coverage and statement "
            "coverage requirements. MC/DC is required to ensure that the most "
            "critical software has been thoroughly tested at the structural level."
        ),
        "keywords": ["MC/DC", "structural coverage", "testing", "Level A", "decision coverage"],
    },
    {
        "rule_id": "DO178B-REQ-002",
        "standard": "DO-178B",
        "section": "6.3.2",
        "dal_level": "B",
        "title": "Decision Coverage for Level B Software",
        "full_text": (
            "For Level B software, structural coverage analysis shall demonstrate "
            "decision coverage. Every point of entry and exit in the program has "
            "been invoked at least once, and every decision in the program has "
            "taken all possible outcomes at least once. Decision coverage is "
            "sufficient for Level B; MC/DC is not required."
        ),
        "keywords": ["decision coverage", "testing", "Level B", "structural coverage"],
    },
    {
        "rule_id": "DO178B-REQ-003",
        "standard": "DO-178B",
        "section": "6.3.3",
        "dal_level": "C",
        "title": "Statement Coverage for Level C Software",
        "full_text": (
            "For Level C software, structural coverage analysis shall demonstrate "
            "statement coverage. Every statement in the program has been invoked "
            "at least once. Statement coverage is the minimum structural coverage "
            "criterion and is sufficient for Level C software."
        ),
        "keywords": ["statement coverage", "testing", "Level C"],
    },
    {
        "rule_id": "DO178B-REQ-004",
        "standard": "DO-178B",
        "section": "5.1",
        "dal_level": "A",
        "title": "Software Requirements Traceability",
        "full_text": (
            "Bidirectional traceability shall be established between system "
            "requirements allocated to software and high-level software requirements, "
            "between high-level requirements and low-level requirements, between "
            "low-level requirements and source code, and between source code and "
            "test cases. This traceability is mandatory for all DAL levels but "
            "is most rigorously enforced at Level A."
        ),
        "keywords": ["traceability", "requirements", "bidirectional", "Level A"],
    },
    {
        "rule_id": "DO178B-REQ-005",
        "standard": "DO-178B",
        "section": "7.2",
        "dal_level": "A",
        "title": "Configuration Management - Problem Reporting",
        "full_text": (
            "A problem reporting process shall be established to document and "
            "track all software anomalies discovered during testing, verification, "
            "and operational use. Each problem report shall include: description "
            "of the problem, affected software version, severity classification, "
            "analysis of impact, and corrective action. For Level A, all problem "
            "reports must be resolved or have approved deferred status before "
            "certification."
        ),
        "keywords": ["configuration management", "problem reporting", "anomalies", "Level A"],
    },
    {
        "rule_id": "DO178B-REQ-006",
        "standard": "DO-178B",
        "section": "4.2",
        "dal_level": "A",
        "title": "Software Development Plan",
        "full_text": (
            "The Software Development Plan (SDP) shall define the software "
            "lifecycle processes, including standards, tools, methods, and "
            "procedures to be used. For Level A software, the SDP must address: "
            "software development environment, programming languages and coding "
            "standards, requirements capture methodology, design methodology, "
            "integration strategy, and the means to ensure compliance with "
            "objectives at each lifecycle stage."
        ),
        "keywords": ["SDP", "development plan", "lifecycle", "Level A", "standards"],
    },
    {
        "rule_id": "DO178B-REQ-007",
        "standard": "DO-178B",
        "section": "6.4.1",
        "dal_level": "A",
        "title": "Independence of Verification Activities",
        "full_text": (
            "For Level A software, verification activities including reviews, "
            "analyses, and test case development shall be performed by personnel "
            "independent of the developer who created the item being verified. "
            "Independence means that the verifier did not develop the software "
            "or write the requirements under review. This independence requirement "
            "applies to reviews of requirements, design, code, and test cases."
        ),
        "keywords": ["independence", "verification", "review", "Level A"],
    },
    {
        "rule_id": "DO178B-REQ-008",
        "standard": "DO-178B",
        "section": "11.1",
        "dal_level": "A",
        "title": "Tool Qualification - Criteria 1 Tools",
        "full_text": (
            "Development tools whose output is part of airborne software and "
            "thus could introduce errors (Criteria 1 tools) shall be qualified "
            "to the same software level as the airborne software they produce. "
            "This includes compilers, linkers, and code generators. Tool "
            "qualification requires demonstration that the tool produces correct "
            "output for its intended use, through tool operational requirements "
            "and tool verification."
        ),
        "keywords": ["tool qualification", "compiler", "criteria 1", "Level A"],
    },
    {
        "rule_id": "DO178B-REQ-009",
        "standard": "DO-178B",
        "section": "5.3",
        "dal_level": "B",
        "title": "Low-Level Requirements Development",
        "full_text": (
            "Low-level requirements shall be developed from high-level requirements "
            "and software architecture. They shall include: detailed algorithmic "
            "behavior, data structures, memory allocation constraints, timing "
            "constraints, and interrupt handling. Low-level requirements must be "
            "verifiable, conformant to standards, traceable to high-level "
            "requirements, and accurate with respect to system requirements."
        ),
        "keywords": ["low-level requirements", "design", "algorithms", "Level B"],
    },
    {
        "rule_id": "DO178B-REQ-010",
        "standard": "DO-178B",
        "section": "12.1",
        "dal_level": "A",
        "title": "Software Life Cycle Data - Compliance Summary",
        "full_text": (
            "The Software Accomplishment Summary (SAS) shall document the "
            "compliance status of the software with the certification plan. "
            "It shall include: a summary of software lifecycle activities, "
            "deviations from plans and their justification, results of "
            "verification activities, open problem reports with impact analysis, "
            "and a statement of compliance with the applicable objectives of "
            "DO-178B for the assigned software level."
        ),
        "keywords": ["SAS", "accomplishment summary", "compliance", "certification", "Level A"],
    },
    # Table A-3: Verification of Outputs of Software Requirements Process
    {
        "rule_id": "DO178B-REQ-011",
        "standard": "DO-178B",
        "section": "Table A-3 Objective 1",
        "dal_level": "A, B, C, D",
        "title": "High-level requirements comply with system requirements",
        "full_text": "High-level requirements (HLR) must be developed and verified to ensure they accurately reflect the system requirements allocated to software.",
        "keywords": ["HLR", "system requirements", "compliance"],
    },
    {
        "rule_id": "DO178B-REQ-012",
        "standard": "DO-178B",
        "section": "Table A-3 Objective 6",
        "dal_level": "A, B, C",
        "title": "HLR Traceability to System Requirements",
        "full_text": "Bi-directional traceability must exist between high-level requirements and the system requirements they satisfy.",
        "keywords": ["traceability", "HLR", "system requirements"],
    },

    # Table A-4: Verification of Outputs of Software Design Process
    {
        "rule_id": "DO178B-REQ-013",
        "standard": "DO-178B",
        "section": "Table A-4 Objective 1",
        "dal_level": "A, B, C",
        "title": "Low-level requirements comply with high-level requirements",
        "full_text": "Low-level requirements (LLR) and the software architecture must comply with and be traceable to the high-level requirements.",
        "keywords": ["LLR", "HLR", "architecture", "design"],
    },
    {
        "rule_id": "DO178B-REQ-014",
        "standard": "DO-178B",
        "section": "Table A-4 Objective 8",
        "dal_level": "A, B, C",
        "title": "Software architecture is consistent",
        "full_text": "The software architecture components must have consistent data flow and control flow relationships.",
        "keywords": ["architecture", "data flow", "control flow"],
    },

    # Table A-5: Verification of Outputs of Software Coding and Integration Process
    {
        "rule_id": "DO178B-REQ-015",
        "standard": "DO-178B",
        "section": "Table A-5 Objective 1",
        "dal_level": "A, B, C",
        "title": "Source Code complies with Low-Level Requirements",
        "full_text": "The Source Code must be verified for compliance with the low-level requirements through review or analysis.",
        "keywords": ["source code", "LLR", "verification"],
    },
    {
        "rule_id": "DO178B-REQ-016",
        "standard": "DO-178B",
        "section": "Table A-5 Objective 2",
        "dal_level": "A, B, C",
        "title": "Source Code complies with Software Architecture",
        "full_text": "The Source Code must be verified for compliance with the software architecture (e.g., partitioning schemes).",
        "keywords": ["source code", "architecture", "partitioning"],
    },
    {
        "rule_id": "DO178B-REQ-017",
        "standard": "DO-178B",
        "section": "Table A-5 Objective 4",
        "dal_level": "A, B, C",
        "title": "Source Code complies with Software Coding Standards",
        "full_text": "Source Code must be reviewed for compliance with defined coding standards (e.g., naming conventions, complexity limits).",
        "keywords": ["coding standards", "naming", "complexity"],
    },

    # Table A-7: Verification of Verification Process Results
    {
        "rule_id": "DO178B-REQ-018",
        "standard": "DO-178B",
        "section": "Table A-7 Objective 5",
        "dal_level": "A, B, C",
        "title": "Statement Coverage Analysis",
        "full_text": "For Level A, B, and C software, every executable statement in the code must be exercised by requirements-based tests.",
        "keywords": ["statement coverage", "testing", "structural analysis"],
    },
    {
        "rule_id": "DO178B-REQ-019",
        "standard": "DO-178B",
        "section": "Table A-7 Objective 7",
        "dal_level": "A",
        "title": "Modified Condition/Decision Coverage (MC/DC)",
        "full_text": "For Level A software, every condition in a decision must be shown to independently affect the output of that decision.",
        "keywords": ["MC/DC", "decision coverage", "Level A"],
    },

    # Table A-9: Software Quality Assurance Process
    {
        "rule_id": "DO178B-REQ-020",
        "standard": "DO-178B",
        "section": "Table A-9 Objective 1",
        "dal_level": "A, B, C, D",
        "title": "QA compliance with plans and standards",
        "full_text": "Assurance is obtained that software development and integral processes comply with approved plans and standards.",
        "keywords": ["QA", "audit", "process assurance"],
    },
    # --- TABLE A-5 OBJECTIVES (Coding) ---
    {
        "rule_id": "DO178B-REQ-021",
        "standard": "DO-178B",
        "section": "Table A-5 Objective 3",
        "dal_level": "A, B, C",
        "title": "Source Code is Verifiable",
        "full_text": "The Source Code must be structured and documented such that it can be verified. This involves avoiding complex or non-deterministic constructs that impede analysis.",
        "keywords": ["verifiable", "complexity", "code analysis"],
    },
    {
        "rule_id": "DO178B-REQ-022",
        "standard": "DO-178B",
        "section": "Table A-5 Objective 5",
        "dal_level": "A, B, C",
        "title": "Source Code Accuracy and Consistency",
        "full_text": "Source Code must be accurate and consistent with the Low-Level Requirements, ensuring no unintended functionality is introduced.",
        "keywords": ["accuracy", "consistency", "unintended functionality"],
    },

    # --- TABLE A-7 OBJECTIVES (Verification) ---
    {
        "rule_id": "DO178B-REQ-023",
        "standard": "DO-178B",
        "section": "Table A-7 Objective 5",
        "dal_level": "A, B",
        "title": "Decision Coverage Analysis",
        "full_text": "Verification must show that every decision in the program has taken all possible outcomes at least once.",
        "keywords": ["decision coverage", "structural coverage", "Level B"],
    },
    {
        "rule_id": "DO178B-REQ-024",
        "standard": "DO-178B",
        "section": "Table A-7 Objective 8",
        "dal_level": "A, B",
        "title": "Data Coupling and Control Coupling",
        "full_text": "Structural coverage analysis shall confirm the requirements-based testing has exercised the data and control coupling between code components.",
        "keywords": ["data coupling", "control coupling", "integration"],
    },

    # --- TABLE A-6 (Integration Testing) ---
    {
        "rule_id": "DO178B-REQ-025",
        "standard": "DO-178B",
        "section": "Table A-6 Objective 1",
        "dal_level": "A, B, C, D",
        "title": "Executable Object Code complies with HLR",
        "full_text": "The Executable Object Code (EOC) must be tested to demonstrate compliance with the High-Level Requirements.",
        "keywords": ["EOC", "testing", "HLR compliance"],
    }        
]
