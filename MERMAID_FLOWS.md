# 🔄 Table Comparison - Mermaid Flow Diagrams

This file contains Mermaid diagrams that visualize the table comparison workflow and architecture.

## 🎯 Overall Process Flow

```mermaid
flowchart TD
    A[📊 Input Tables<br/>df1 & df2] --> B{Select Method}
    
    B -->|Fast| C[📛 Name-Based<br/>Embeddings]
    B -->|Comprehensive| D[📊 Content-Based<br/>Embeddings]
    B -->|Balanced| E[🔀 Hybrid<br/>Embeddings]
    B -->|Maximum Accuracy| F[🎯 Ensemble<br/>Method]
    
    C --> G[🧠 Ollama<br/>mxbai-embed-large]
    D --> G
    E --> G
    
    G --> H[📐 Cosine<br/>Similarity]
    
    F --> I[📊 Statistical<br/>Analysis]
    F --> J[🔍 Pattern<br/>Detection]
    F --> K[🔤 String<br/>Matching]
    F --> H
    
    I --> L[📈 Weighted<br/>Combination]
    J --> L
    K --> L
    H --> L
    
    L --> M[🎯 Best Matches<br/>High Confidence]
    M --> N[📊 Visualizations<br/>& Reports]
    
    style A fill:#e1f5fe
    style G fill:#f3e5f5
    style M fill:#e8f5e8
    style N fill:#fff3e0
```

## 🧠 Embedding Approaches Comparison

```mermaid
graph LR
    subgraph "📛 Name-Based"
        A1[Column Names] --> A2[first_name, email] --> A3[Ollama Embeddings] --> A4[Vector: 1024 dims]
    end
    
    subgraph "📊 Content-Based"
        B1[Column + Data] --> B2[first_name: Text column<br/>150 unique values<br/>examples: John, Mary...] --> B3[Ollama Embeddings] --> B4[Vector: 1024 dims]
    end
    
    subgraph "🔀 Hybrid"
        C1[Name + Content] --> C2[Weighted Sum<br/>40% name + 60% content] --> C3[Combined Vector] --> C4[Vector: 1024 dims]
    end
    
    A4 --> D[Similarity Matrix]
    B4 --> D
    C4 --> D
    
    D --> E[📊 Comparison Results]
    
    style A1 fill:#e3f2fd
    style B1 fill:#e8f5e8
    style C1 fill:#fff3e0
    style D fill:#f3e5f5
    style E fill:#fce4ec
```

## 🎯 Ensemble Method Architecture

```mermaid
flowchart TD
    A[📊 Input DataFrames] --> B[🔀 Multi-Method Analysis]
    
    B --> C[📛 Semantic<br/>Embeddings<br/>Weight: 35%]
    B --> D[🔤 Fuzzy String<br/>Matching<br/>Weight: 25%]
    B --> E[📊 Statistical<br/>Signatures<br/>Weight: 25%]
    B --> F[🔍 Content-Based<br/>Patterns<br/>Weight: 15%]
    
    C --> G[🧠 Ollama<br/>mxbai-embed-large]
    D --> H[🔤 Levenshtein<br/>Distance]
    E --> I[📊 Data Type<br/>Distribution]
    F --> J[🔍 Regex Pattern<br/>Detection]
    
    G --> K[📐 Cosine Similarity]
    H --> L[🔢 String Similarity]
    I --> M[📊 Statistical Score]
    J --> N[🔍 Pattern Score]
    
    K --> O[⚖️ Weighted<br/>Ensemble]
    L --> O
    M --> O
    N --> O
    
    O --> P{🎯 Confidence<br/>Threshold?}
    P -->|> 0.7| Q[✅ High Confidence<br/>Matches]
    P -->|< 0.7| R[⚠️ Low Confidence<br/>Review Needed]
    
    Q --> S[📈 Final Results<br/>& Visualizations]
    R --> S
    
    style A fill:#e1f5fe
    style O fill:#f3e5f5
    style Q fill:#e8f5e8
    style R fill:#ffebee
    style S fill:#fff3e0
```

## 🔍 Data Content Analysis Pipeline

```mermaid
flowchart LR
    A[📊 Column Data] --> B{Data Type<br/>Detection}
    
    B -->|Boolean| C[📊 True/False<br/>Distribution]
    B -->|Text| D[📝 Sample Values<br/>& Patterns]
    B -->|Numeric| E[📈 Statistics<br/>& Distribution]
    B -->|DateTime| F[📅 Date Range<br/>& Format]
    
    C --> G[Boolean column: 75 True, 75 False, 50% true rate]
    D --> H[Text column: 150 unique values,<br/>avg length 23.4,<br/>examples: john@email.com...]
    E --> I[Numerical column: range 30000 to 150000,<br/>mean 89000, std 35000,<br/>examples: 45000, 67000...]
    F --> J[Date column: range 3650 days<br/>from 2014-01-01 to 2024-01-01,<br/>samples: 2018-05-15...]
    
    G --> K[🧠 Rich Content<br/>Embedding]
    H --> K
    I --> K
    J --> K
    
    K --> L[📊 Content-Based<br/>Similarity Matrix]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style K fill:#e8f5e8
    style L fill:#fff3e0
```

## 📊 Performance & Method Selection Flow

```mermaid
flowchart TD
    A[🎯 Use Case Analysis] --> B{Primary Goal?}
    
    B -->|Speed| C[⚡ Fast Processing<br/>< 2 seconds]
    B -->|Accuracy| D[🎯 High Accuracy<br/>Maximum matches]
    B -->|Balance| E[⚖️ Speed + Accuracy<br/>Production ready]
    
    C --> F[📛 Name-Based<br/>Embeddings]
    D --> G[📊 Content-Based<br/>Embeddings]
    E --> H[🔀 Hybrid<br/>Approach]
    
    F --> I[18 high confidence<br/>0.701 avg similarity<br/>~2 seconds]
    G --> J[25 high confidence<br/>0.934 avg similarity<br/>~6 seconds]
    H --> K[25 high confidence<br/>0.888 avg similarity<br/>~8 seconds]
    
    I --> L{Acceptable<br/>Results?}
    J --> L
    K --> L
    
    L -->|Yes| M[✅ Deploy Solution]
    L -->|No| N[🎯 Try Ensemble<br/>Method]
    
    N --> O[🎯 All Methods<br/>Combined]
    O --> P[📊 Maximum<br/>Accuracy]
    P --> M
    
    style A fill:#e1f5fe
    style M fill:#e8f5e8
    style N fill:#fff3e0
    style P fill:#f3e5f5
```

## 🔧 Technical Architecture

```mermaid
graph TB
    subgraph "🎯 User Interface"
        A[Jupyter Notebook] --> B[Python Functions]
    end
    
    subgraph "🧠 AI/ML Layer"
        C[Ollama Server] --> D[mxbai-embed-large]
        C --> E[nomic-embed-text]
        C --> F[bge-base-en-v1.5]
    end
    
    subgraph "🔧 Processing Engine"
        G[Embedding Generator] --> H[Similarity Calculator]
        I[Statistical Analyzer] --> H
        J[Pattern Detector] --> H
        K[String Matcher] --> H
    end
    
    subgraph "📊 Data Layer"
        L[Pandas DataFrames] --> M[Column Metadata]
        L --> N[Sample Data]
        L --> O[Statistical Profiles]
    end
    
    subgraph "📈 Output Layer"
        P[Similarity Matrices] --> Q[Match Results]
        Q --> R[Visualizations]
        Q --> S[Performance Metrics]
    end
    
    B --> G
    B --> I
    B --> J
    B --> K
    
    G --> C
    
    L --> G
    L --> I
    L --> J
    L --> K
    
    H --> P
    
    style A fill:#e3f2fd
    style C fill:#f3e5f5
    style G fill:#e8f5e8
    style L fill:#fff3e0
    style P fill:#fce4ec
```

## 🎨 Visualization Pipeline

```mermaid
flowchart LR
    A[📊 Similarity Results] --> B[📈 Visualization<br/>Generator]
    
    B --> C[🔥 Similarity<br/>Heatmap]
    B --> D[📊 Method<br/>Comparison]
    B --> E[🥧 Agreement<br/>Analysis]
    B --> F[📋 Results<br/>Table]
    
    C --> G[30x24 inches<br/>Color-coded matrix<br/>All column pairs]
    D --> H[Bar charts<br/>Performance metrics<br/>Speed vs Accuracy]
    E --> I[Pie chart<br/>Method consensus<br/>Agreement rates]
    F --> J[Detailed table<br/>Top matches<br/>Confidence scores]
    
    G --> K[📱 Interactive<br/>Display]
    H --> K
    I --> K
    J --> K
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style K fill:#e8f5e8
```

## 🔄 End-to-End Workflow

```mermaid
flowchart TD
    Start([🚀 Start Process]) --> A[📥 Load Tables<br/>df1 & df2]
    
    A --> B[🔍 Analyze<br/>Column Structure]
    B --> C{Column Names<br/>Descriptive?}
    
    C -->|Yes| D[📛 Use Name-Based<br/>Fast & Efficient]
    C -->|No| E[📊 Use Content-Based<br/>Deep Analysis]
    C -->|Mixed| F[🔀 Use Hybrid<br/>Best of Both]
    
    D --> G[🧠 Generate<br/>Embeddings]
    E --> G
    F --> G
    
    G --> H[📐 Calculate<br/>Similarities]
    H --> I{Confidence<br/>Acceptable?}
    
    I -->|Yes| J[✅ Generate<br/>Final Report]
    I -->|No| K[🎯 Try Ensemble<br/>Method]
    
    K --> L[🔧 Multi-Method<br/>Analysis]
    L --> M[⚖️ Weighted<br/>Combination]
    M --> J
    
    J --> N[📊 Create<br/>Visualizations]
    N --> O[📋 Export<br/>Results]
    O --> End([🎉 Complete])
    
    style Start fill:#e8f5e8
    style G fill:#f3e5f5
    style J fill:#e8f5e8
    style End fill:#e8f5e8
```

---


These diagrams provide visual documentation for:
- 👥 **Stakeholders**: Understanding capabilities
- 👨‍💻 **Developers**: Implementation guidance
- 📚 **Documentation**: Technical specifications
- 🎓 **Training**: Educational materials
