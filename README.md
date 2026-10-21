# Solver ARC-AGI-1 com Gemini

Solver simples para o ARC-AGI-1 e um dashboard em Streamlit que compara níveis de
reasoning e modelos. Pergunta central: mais reasoning resolve mais tarefas, e a que custo?

A versão publicada é **só para consulta**. Os experimentos são rodados localmente e os
resultados (`results/`) são enviados ao repositório; é isso que o app publicado mostra.

## Arquivos

| Arquivo | Função |
|---|---|
| `main.py` | Ponto de entrada: uma tarefa ou um arquivo de tarefas |
| `prompt.py` | Monta o prompt |
| `llm.py` | Chamadas à API (Google padrão, OpenRouter opcional), retry e troca de chave |
| `validator.py` | Extrai a grade da resposta e compara com a esperada |
| `dashboard.py`, `flashback.py` | Dashboard (painel, flashback, comparação de modelos) |
| `data/` | Tarefas do ARC-AGI-1 (JSON) |
| `tasks*.txt` | Listas de tarefas (uma por linha) |
| `results/` | Resultados dos experimentos (`live_*.jsonl`, `summary_*.json`) |

## Rodar localmente

```
uv sync
```

Crie um `.env` (não é versionado):

```
GOOGLE_API_KEY=...
# GOOGLE_API_KEY_2=..., GOOGLE_API_KEY_3=...   opcionais: usadas quando a anterior recebe 429/401/403
ARC_LOCAL_MODE=1        # libera executar e excluir experimentos no dashboard local
```

Linha de comando:

```
uv run main.py 3aa6fb7a --reasoning low
uv run main.py --file tasks.txt --label meu_exp --reasoning minimal low medium high --wait 30
```

Dashboard (com `--server.address localhost`, ninguém da rede acessa o seu dashboard):

```
uv run streamlit run dashboard.py --server.address localhost
```

O dashboard só mostra os controles de execução e exclusão quando `ARC_LOCAL_MODE=1` está no
`.env` **e** o navegador está na mesma máquina. Em qualquer outro caso (nuvem, outro computador
da rede) ele abre só para consulta. As chaves do `.env` aparecem mascaradas; dá para adicionar
chaves extras só para a sessão atual. Os arquivos de tarefas são escolhidos numa lista dos
`tasks*.txt` da pasta do projeto.

## Publicar no Streamlit Community Cloud

1. Rode os experimentos localmente e faça commit de `results/live_*.jsonl` e
   `results/summary_*.json`.
2. Envie o repositório ao GitHub. O `.gitignore` já exclui `.env`, `.venv` e os logs.
3. Em <https://share.streamlit.io>: **Create app** > repositório, branch e `dashboard.py`.
4. **Advanced settings** > **Python version: 3.13** (o projeto exige `>=3.13`; o padrão da
   nuvem é 3.12). Não configure Secrets: o app publicado não usa chaves.

A nuvem inicia o app a partir da **raiz do repositório** e só lê o `.streamlit/config.toml` da
raiz. Por isso esta pasta deve ser a raiz do repositório. Para atualizar o que o app mostra,
basta fazer commit de novos resultados.
