# Sistema RAG híbrido con Pinecone

Módulo de recuperación en Python 3.12 + LangChain que indexa documentación técnica en
un índice **Pinecone Serverless** y la recupera con un **recuperador híbrido**:
similitud vectorial (Pinecone) + búsqueda léxica **BM25**, fusionadas con un
`EnsembleRetriever`. Un script de evaluación mide **Recall@5** y **Precision@5** sobre
un golden set de 5 preguntas.

Pre-entrega 4 del curso AI Engineering: "Sistema RAG escalable en la nube con Pinecone".

## Estructura

```
config.py        # settings desde .env, embeddings de 1536 dims, cliente de Pinecone
init_index.py    # crea el índice serverless solo si no existe (idempotente)
corpus.py        # carga Markdown/JSON de data/ y chunking con RecursiveCharacterTextSplitter
ingest.py        # embeddings + upsert por lotes en un namespace, con el texto en la metadata
rag_system.py    # clase RAGSystem: EnsembleRetriever(Pinecone + BM25) -> top-5
metrics.py       # Recall@k, Precision@k y precisión máxima alcanzable (funciones puras)
evaluate.py      # benchmark con golden_set.json, imprime el resumen
golden_set.json  # 5 preguntas con su documento_id_esperado
data/            # 7 documentos de la librería ficticia "Caudal" (6 .md + 1 .json)
docs/            # salidas reales: init_index, ingesta, consulta de ejemplo y evaluación
tests/           # pytest sin red: Pinecone falso, embeddings falsos, BM25 real
```

## El flujo

```
data/*.md, data/*.json
  -> corpus.load_documents()     una Document por sección ("página"), con doc_id, source,
                                 page, section, category, title
  -> corpus.split_documents()    RecursiveCharacterTextSplitter (700 tokens, solapamiento 100)
                                 chunk_id determinista: "<doc_id>-p<página>-c<n>"
  -> ingest.upsert_chunks()      OpenAIEmbeddings (1536 dims) -> index.upsert(namespace=...)
                                 metadata: doc_id, chunk_id, source, page, section,
                                 category, title, chunk_index y text (el chunk original)

consulta
  -> RAGSystem.retrieve()
       ├─ PineconeVectorStore.as_retriever(k=5)   semántico, en el namespace configurado
       └─ BM25Retriever(k=5)                      léxico, en memoria
     EnsembleRetriever(weights=[0.5, 0.5], id_key="chunk_id")   Reciprocal Rank Fusion
  -> top-5 chunks (Document con texto y metadata)
```

## Instalación

Requiere Python 3.12 y [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Variables de entorno

Copiá `.env.example` a `.env` y completalo. El `.env` está en `.gitignore`: nunca se sube.

| Variable | Obligatoria | Default | Para qué |
|---|---|---|---|
| `PINECONE_API_KEY` | Sí | | Key de Pinecone |
| `INDEX_NAME` | No | `pre-entrega-4` | Nombre del índice serverless |
| `PINECONE_NAMESPACE` | No | `caudal-docs` | Namespace donde van todos los vectores |
| `OPENAI_API_KEY` | Sí | | Key de OpenAI, o de Gemini si usás `OPENAI_BASE_URL` |
| `OPENAI_BASE_URL` | No | | Endpoint compatible con OpenAI (p. ej. el de Gemini) |
| `EMBEDDING_MODEL` | No | `gemini-embedding-001` con `OPENAI_BASE_URL`, si no `text-embedding-3-small` | Modelo de embeddings |

Con cualquiera de los dos modelos se piden **1536 dimensiones** (`dimensions=1536`):
`text-embedding-3-small` las da de forma nativa y `gemini-embedding-001` acepta reducir
su salida. Con Gemini además se desactiva `check_embedding_ctx_length`, porque ese
chequeo tokeniza con tiktoken y manda ids de tokens, que el endpoint de Gemini no acepta.

## Cómo replicar el índice

```bash
uv sync
cp .env.example .env              # y completá PINECONE_API_KEY y OPENAI_API_KEY
uv run python init_index.py       # crea el índice serverless (aws / us-east-1, 1536, cosine)
uv run python ingest.py --reset   # chunking + embeddings + upsert en el namespace
uv run python rag_system.py "¿Qué significa el error CAU-417?"   # consulta de prueba
uv run python evaluate.py         # métricas (híbrido, solo vector y solo BM25)
```

- `init_index.py` es idempotente: la primera vez crea el índice y espera a que esté
  listo; las siguientes solo verifica que exista con la **misma dimensión y métrica**
  (si no, falla con un mensaje claro en vez de esperar al primer upsert).
  Salida real de dos corridas seguidas: [`docs/init_index.txt`](docs/init_index.txt).
- `ingest.py` también llama a `ensure_index`, así que alcanza con correr la ingesta.
  Volver a correrla no duplica nada (los ids son deterministas, el upsert sobreescribe);
  `--reset` vacía el namespace antes, útil si cambiás el chunking. Al final espera a que
  los vectores sean visibles (Pinecone es eventualmente consistente).
  Salida real: [`docs/ingesta.txt`](docs/ingesta.txt).
- Región `us-east-1` en AWS porque es la que admite el plan gratuito de Pinecone.

## Decisiones

**Dataset.** Siete documentos de una librería de Python ficticia, *Caudal*
(procesamiento de flujos): instalación, API de flujos, conectores, errores,
rendimiento, seguridad (Markdown con front matter `doc_id`/`titulo`/`categoria`) y un
changelog en JSON. Es ficticia a propósito: el modelo de embeddings no la conoce de
antemano, y tiene muchos nombres propios y códigos (`CAU-417`, `KafkaFuente`,
`caudal-prof`, `tamano_lote`) donde la búsqueda léxica tiene que aportar.

**Página = sección.** No hay PDFs: una "página" es una sección `## ` del Markdown o un
elemento de `secciones` del JSON. Viaja en la metadata como `page` (y el título en
`section`), igual que el número de página de un PDF.

**Chunking.** `RecursiveCharacterTextSplitter` con `chunk_size=700` y
`chunk_overlap=100` medidos **en tokens** (tiktoken `cl100k_base`), dentro del rango
de 500-800 que sugiere la consigna. Se parte por sección, así que ningún chunk mezcla
dos temas; con este dataset cada sección entra en un chunk: **21 chunks de 229 a 539
tokens**. Una sección más larga se partiría en varios (`-c0`, `-c1`, ...).

**Ingesta con el SDK nativo.** `index.upsert` en lotes de 100 en lugar de
`PineconeVectorStore.add_documents`, para controlar ids y metadata exactos. La lectura
sí usa `PineconeVectorStore` (lee el texto de la clave `text` de la metadata).

**Namespace.** Todo va a `PINECONE_NAMESPACE` (`caudal-docs`), nunca al namespace por
defecto: el mismo índice puede alojar otros corpus sin mezclarse, y `--reset` borra
solo lo nuestro.

**BM25 en memoria, reconstruido desde los archivos locales.** Pinecone no hace BM25, así
que el `BM25Retriever` necesita los chunks en memoria. En vez de persistirlos en un JSON
aparte, `RAGSystem` los reconstruye con la **misma función de chunking** que la ingesta.
Como el chunking es determinista, los `chunk_id` coinciden con los de Pinecone y el
`EnsembleRetriever` (`id_key="chunk_id"`) fusiona un mismo chunk encontrado por las dos
vías en un único resultado. Contra: si cambiás `data/` sin reingestar, BM25 y Pinecone
quedan desfasados (por eso `ingest.py --reset`). A escala real convendría guardar los
chunks en un almacén compartido o usar los sparse vectors de Pinecone.

**Tokenizador de BM25 para español.** Minúsculas, sin tildes, sin stopwords, y conserva
como un solo token las palabras con guion o guion bajo (`cau-417`, `caudal-prof`,
`tamano_lote`), que son justo las que más discriminan.

**Fusión.** `EnsembleRetriever` con pesos 0.5/0.5 (RRF, `c=60`). Cada recuperador trae
5; la unión puede tener hasta 10 y nos quedamos con los 5 mejores.

**Metadata numérica.** Pinecone devuelve los números de la metadata como float
(`page=1.0`); `RAGSystem` normaliza `page` y `chunk_index` a int para que los resultados
de Pinecone y de BM25 tengan la misma forma.

## Métricas

Definiciones (en `metrics.py`), por pregunta y sobre los 5 chunks devueltos:

- **Recall@5**: 1 si el `documento_id_esperado` aparece entre los 5 chunks, 0 si no
  ("¿está el documento correcto entre los 5?").
- **Precision@5**: fracción de los 5 chunks que son **útiles**. Un chunk es útil si su
  `doc_id` está entre los documentos relevantes de la pregunta: por defecto solo el
  esperado; el golden set puede listar más con `documentos_relevantes` (la pregunta 4,
  sobre la versión de Python, también la responde el changelog).
- **Precisión máxima alcanzable**: cada documento tiene 3 chunks, así que si hay un solo
  documento relevante ni un recuperador perfecto pasa de 3/5 = 0.60. Se informa al lado
  para leer la precisión en contexto.

### Resultados reales

Índice `pre-entrega-4`, namespace `caudal-docs`, 21 chunks, embeddings
`gemini-embedding-001` (1536 dims). Salida completa, pregunta por pregunta:
[`docs/evaluacion.txt`](docs/evaluacion.txt).

| Modo | Recall@5 | Precision@5 | Precisión máx. alcanzable |
|---|---|---|---|
| **Híbrido (vector + BM25)** | **1.00** | **0.52** | 0.68 |
| Solo vector (Pinecone) | 1.00 | 0.52 | 0.68 |
| Solo BM25 | 1.00 | 0.36 | 0.68 |

Lectura:

- Los tres modos encuentran siempre el documento correcto (Recall@5 = 1.00). Con 7
  documentos y 5 preguntas bien diferenciadas, el recall no discrimina.
- La diferencia está en la precisión: BM25 solo trae más chunks de otros documentos que
  comparten palabras sueltas ("checkpoint", "flujo"). El híbrido alcanza 0.52 sobre un
  techo de 0.68 (76 % del máximo posible).
- En este corpus el híbrido **empata** con el vector solo, pregunta por pregunta: los
  embeddings de Gemini ya resuelven bien estas preguntas y RRF solo cambia el orden
  dentro del top-5 (en P5, por ejemplo, el híbrido sube un chunk de la API de flujos que
  BM25 puso primero por la palabra "checkpoints"). O sea: con este golden set, BM25 no
  le suma precisión al vector, pero tampoco se la resta, y aporta robustez para
  consultas con códigos exactos (`CAU-417`, `caudal-prof`) que un modelo de embeddings
  puede no distinguir. Consulta de ejemplo: [`docs/consulta_ejemplo.txt`](docs/consulta_ejemplo.txt).
- Cinco preguntas son pocas para sacar conclusiones estadísticas: la evaluación sirve
  como prueba de humo y como base para comparar cambios (pesos del ensemble, tamaño de
  chunk, otro modelo de embeddings).

```bash
uv run python evaluate.py --modos hibrido            # solo el híbrido
uv run python evaluate.py --salida docs/evaluacion.txt
```

## Tests

```bash
uv run pytest -q
```

Sin red y sin keys: `DeterministicFakeEmbedding` en lugar del modelo real, un
`FakePinecone`/`FakeIndex` en memoria (con búsqueda por coseno real y que, como
Pinecone, devuelve los números como float) y `BM25Retriever` real. Cubren:

- creación condicional del índice (no existe -> lo crea serverless aws/us-east-1 con
  1536/cosine; existe -> no lo recrea; dimensión o métrica distinta -> error);
- carga de Markdown y JSON, metadata y ids estables del chunking;
- registros del upsert con `text`, `source`, `page`, `category`, `doc_id` y `chunk_id`,
  lotes, namespace (nada en el namespace por defecto) e idempotencia;
- `RAGSystem`: devuelve 5 sin duplicados, combina resultados de Pinecone y de BM25,
  consulta en el namespace configurado;
- métricas con casos calculados a mano y `evaluate()` con un RAG de juguete.

## Limitaciones

- BM25 vive en memoria: con millones de chunks habría que reemplazarlo por un motor
  léxico externo o por los sparse vectors de Pinecone (búsqueda híbrida nativa).
- `langchain-community` (donde vive `BM25Retriever`) avisa que está en retirada; el
  aviso se silencia en los tests.
- El sistema solo recupera: no genera respuestas con un LLM (la consigna lo deja como
  opcional).
