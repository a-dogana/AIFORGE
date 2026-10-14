/**
 * AIFORGE Web Dashboard - Alpine.js Application
 * Reactive single-state architecture without external build tools.
 */

function aiforgeApp() {
    return {
        tab: 'dashboard',
        serverStatus: 'offline',
        statTotalExperiments: 0,
        activeModel: 'BASE:Qwen/Qwen2.5-Coder-0.5B-Instruct',
        serverMsg: '',
        serverMsgColor: 'text-slate-500',
        experiments: [],
        selectedExperiment: null,

        // Chat Playground
        chatMessages: [
            { sender: 'ai', text: 'Ciao! Sono il tuo modello locale specializzato. Chiedimi di scrivere o spiegare codice C o qualsiasi altro algoritmo.' }
        ],
        chatPrompt: '',
        chatMaxTokens: 120,
        chatTemperature: 0.3,
        chatLatency: '',
        chatError: '',
        isGenerating: false,

        // Model Merge Lab
        merge: {
            modelA: '',
            modelB: '',
            strategy: 'slerp',
            alpha: 0.50,
            isPreflighting: false,
            preflight: null,
            isMerging: false,
            lastMergedId: null,
            lastModelA: null,
            lastModelB: null,
            mergeStatusText: '',
            evalLoading: false,
            evalOutputHtml: '',
            ollamaLoading: false,
            ollamaOutputHtml: '',
            ggufLoading: false,
            ggufOutputHtml: ''
        },

        // Forge (Training)
        forge: {
            name: '',
            model: 'Qwen/Qwen2.5-Coder-0.5B-Instruct',
            topic: '',
            method: 'lima',
            turnType: 'single_turn',
            targetSamples: 100,
            qlora: true,
            isRunning: false,
            progress: '0%',
            terminalLogs: ['> Forgia pronta (Engine: Soup Backend). Seleziona la metodologia (LIMA o Rapido) e avvia...']
        },

        // Connectors
        connectors: [],
        newConn: {
            id: '',
            name: '',
            base_url: 'http://127.0.0.1:11434/v1',
            api_key: ''
        },
        connMsg: '',
        isSavingConn: false,
        testingConnId: null,
        connTestResults: {},

        async init() {
            await this.checkStatus();
            await this.loadExperiments();
            await this.loadConnectors();

            // Background polling for server status
            setInterval(() => this.checkStatus(), 4000);
        },

        switchTab(target) {
            this.tab = target;
            if (target === 'experiments') this.loadExperiments();
            if (target === 'connectors') this.loadConnectors();
        },

        // --- Server Status & Lifecycle ---
        async checkStatus() {
            try {
                const res = await fetch('/api/status');
                if (!res.ok) return;
                const data = await res.json();
                this.serverStatus = data.llm_server === 'online' ? 'online' : 'offline';
                if (data.experiments_count !== undefined) {
                    this.statTotalExperiments = data.experiments_count;
                }
            } catch (e) {
                this.serverStatus = 'offline';
            }
        },

        async startServer(targetModel) {
            const expId = targetModel || this.activeModel;
            if (!expId) {
                alert("Seleziona prima un modello da avviare.");
                return;
            }
            this.serverMsg = "Avvio del server in corso...";
            this.serverMsgColor = "text-amber-600";
            try {
                const res = await fetch('/api/server/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ experiment_id: expId })
                });
                if (res.ok) {
                    this.serverMsg = "Server avviato con successo!";
                    this.serverMsgColor = "text-emerald-600";
                    setTimeout(() => this.checkStatus(), 1000);
                } else {
                    const data = await res.json();
                    this.serverMsg = "Errore: " + (data.detail || "Impossibile avviare");
                    this.serverMsgColor = "text-rose-600";
                }
            } catch (e) {
                this.serverMsg = "Errore di rete nell'avvio del server.";
                this.serverMsgColor = "text-rose-600";
            }
        },

        async stopServer() {
            this.serverMsg = "Arresto del server in corso...";
            this.serverMsgColor = "text-slate-500";
            try {
                const res = await fetch('/api/server/stop', { method: 'POST' });
                if (res.ok) {
                    this.serverStatus = 'offline';
                    this.serverMsg = "Server arrestato.";
                    this.serverMsgColor = "text-slate-500";
                }
            } catch (e) {
                console.error("Errore stop server", e);
            }
        },

        toggleServer() {
            if (this.serverStatus === 'online') {
                this.stopServer();
            } else {
                this.startServer(this.activeModel);
            }
        },

        // --- Experiments ---
        async loadExperiments() {
            try {
                const res = await fetch('/api/experiments');
                if (!res.ok) return;
                this.experiments = await res.json();
                this.statTotalExperiments = this.experiments.length;

                // Auto-populate merge selects if empty
                if (this.experiments.length > 0) {
                    if (!this.merge.modelA) this.merge.modelA = this.experiments[0].id;
                    if (!this.merge.modelB) this.merge.modelB = this.experiments[Math.min(1, this.experiments.length - 1)].id;
                }
            } catch (e) {
                console.error("Errore caricamento esperimenti", e);
            }
        },

        inspectExperiment(exp) {
            this.selectedExperiment = exp;
        },

        // --- Chat Playground ---
        async sendChat() {
            const text = this.chatPrompt.trim();
            if (!text || this.isGenerating) return;

            this.chatMessages.push({ sender: 'user', text: text });
            this.chatPrompt = '';
            this.isGenerating = true;
            this.chatError = '';
            this.chatLatency = 'Elaborazione in corso...';

            const startTime = performance.now();
            this.chatMessages.push({ sender: 'ai', text: 'Elaborazione sulla GPU Apple Silicon...' });
            const aiMessageIndex = this.chatMessages.length - 1;

            try {
                const res = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        prompt: text,
                        max_new_tokens: parseInt(this.chatMaxTokens) || 120,
                        temperature: parseFloat(this.chatTemperature) || 0.3
                    })
                });

                const data = await res.json();
                const latency = Math.round(performance.now() - startTime);

                if (res.ok) {
                    this.chatMessages[aiMessageIndex].text = data.text || data.generated_text || "Nessun output generato.";
                    this.chatLatency = `${latency} ms • ${data.tokens_generated || 'N/A'} tok`;
                } else {
                    this.chatMessages[aiMessageIndex].text = "Errore: " + (data.detail || "Generazione non riuscita");
                    this.chatError = data.detail || "Errore del server";
                }
            } catch (e) {
                this.chatMessages[aiMessageIndex].text = "Errore di connessione: assicurati che il server LLM sia avviato (porta 8001).";
                this.chatError = e.message;
            } finally {
                this.isGenerating = false;
                this.$nextTick(() => {
                    const el = document.getElementById('chat-messages-container');
                    if (el) el.scrollTop = el.scrollHeight;
                });
            }
        },

        clearChat() {
            this.chatMessages = [
                { sender: 'ai', text: 'Cronologia chat cancellata. Fai una nuova domanda al modello locale.' }
            ];
            this.chatLatency = '';
            this.chatError = '';
        },

        // --- Model Merge Lab ---
        async runPreflight() {
            if (!this.merge.modelA || !this.merge.modelB) {
                alert("Seleziona sia il Modello A che il Modello B.");
                return;
            }
            this.merge.isPreflighting = true;
            this.merge.preflight = null;
            try {
                const res = await fetch('/api/merge/preflight', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model_a_id: this.merge.modelA,
                        model_b_id: this.merge.modelB,
                        strategy: this.merge.strategy,
                        alpha: parseFloat(this.merge.alpha)
                    })
                });
                const data = await res.json();
                if (!res.ok) {
                    this.merge.preflight = {
                        status: 'INCOMPATIBLE',
                        incompatibilities: [data.detail || "Errore nella verifica di compatibilità"],
                        warnings: []
                    };
                } else {
                    this.merge.preflight = data;
                }
            } catch (e) {
                this.merge.preflight = {
                    status: 'INCOMPATIBLE',
                    incompatibilities: ["Errore di connessione: " + e.message],
                    warnings: []
                };
            } finally {
                this.merge.isPreflighting = false;
            }
        },

        async executeMerge() {
            if (!this.merge.modelA || !this.merge.modelB) return;
            this.merge.isMerging = true;
            this.merge.mergeStatusText = "Fusione dei modelli in corso... Attendere...";
            this.merge.lastMergedId = null;
            this.merge.evalOutputHtml = '';
            this.merge.ollamaOutputHtml = '';

            try {
                const res = await fetch('/api/merge/execute', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model_a_id: this.merge.modelA,
                        model_b_id: this.merge.modelB,
                        strategy: this.merge.strategy,
                        alpha: parseFloat(this.merge.alpha),
                        output_name: `Merged_${this.merge.strategy}_${Date.now().toString().slice(-4)}`
                    })
                });
                const data = await res.json();
                if (res.ok) {
                    this.merge.lastMergedId = data.experiment_id;
                    this.merge.lastModelA = this.merge.modelA;
                    this.merge.lastModelB = this.merge.modelB;
                    this.merge.mergeStatusText = `Merge completato con successo! Nuovo ID: ${data.experiment_id}`;
                    await this.loadExperiments();
                } else {
                    this.merge.mergeStatusText = `Errore merge: ${data.detail || "Operazione fallita"}`;
                }
            } catch (e) {
                this.merge.mergeStatusText = `Errore di rete: ${e.message}`;
            } finally {
                this.merge.isMerging = false;
            }
        },

        async runEval() {
            if (!this.merge.lastMergedId) return;
            this.merge.evalLoading = true;
            this.merge.evalOutputHtml = "Calcolo Perplexity e Cross-Entropy in corso su validation slice...";

            try {
                const res = await fetch('/api/merge/evaluate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model_a_id: this.merge.lastModelA,
                        model_b_id: this.merge.lastModelB,
                        merged_model_id: this.merge.lastMergedId
                    })
                });
                const data = await res.json();
                if (res.ok) {
                    const badgeClass = data.release_gate === 'PROMOTED' ? 'text-emerald-700 font-bold' : 'text-amber-700 font-bold';
                    let html = `[Release Gate]: <span class="${badgeClass}">${data.release_gate}</span>\n`;
                    if (data.metrics && data.metrics.Merged) {
                        const m = data.metrics.Merged;
                        if (m.perplexity) html += `Perplexity Fuso: ${m.perplexity.toFixed(2)}\n`;
                        if (m.accuracy !== undefined) html += `Accuratezza: ${(m.accuracy * 100).toFixed(1)}%\n`;
                    }
                    this.merge.evalOutputHtml = html;
                } else {
                    this.merge.evalOutputHtml = `Errore valutazione: ${data.detail || data.error}`;
                }
            } catch (e) {
                this.merge.evalOutputHtml = `Errore di rete: ${e.message}`;
            } finally {
                this.merge.evalLoading = false;
            }
        },

        async exportOllama() {
            if (!this.merge.lastMergedId) return;
            this.merge.ollamaLoading = true;
            this.merge.ollamaOutputHtml = "Generazione Modelfile per Ollama in corso...";

            try {
                const res = await fetch('/api/export/ollama', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ experiment_id: this.merge.lastMergedId })
                });
                const data = await res.json();
                if (res.ok) {
                    this.merge.ollamaOutputHtml = `✅ Modelfile creato in: <code>${data.modelfile_path}</code><br>Per registrarlo in Ollama, esegui:<br><pre class="mt-1 p-2 bg-slate-900 text-slate-100 rounded text-xs">${data.create_command}</pre>`;
                } else {
                    this.merge.ollamaOutputHtml = `Errore export: ${data.detail}`;
                }
            } catch (e) {
                this.merge.ollamaOutputHtml = `Errore di rete: ${e.message}`;
            } finally {
                this.merge.ollamaLoading = false;
            }
        },

        async exportGGUF() {
            if (!this.merge.lastMergedId) return;
            this.merge.ggufLoading = true;
            this.merge.ggufOutputHtml = "Generazione pacchetto GGUF (Q4_K_M) per LM Studio / llama.cpp in corso...";

            try {
                const res = await fetch('/api/export/gguf', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        experiment_id: this.merge.lastMergedId,
                        quantization: "Q4_K_M"
                    })
                });
                const data = await res.json();
                if (res.ok) {
                    this.merge.ggufOutputHtml = `✅ Pacchetto GGUF creato in: <code>${data.export_dir}</code><br>Target GGUF: <code>${data.target_gguf_path}</code><br>Script di quantizzazione: <code>${data.script_path}</code>`;
                } else {
                    this.merge.ggufOutputHtml = `Errore export GGUF: ${data.detail || "Fallito"}`;
                }
            } catch (e) {
                this.merge.ggufOutputHtml = `Errore di rete: ${e.message}`;
            } finally {
                this.merge.ggufLoading = false;
            }
        },

        // --- Forge (Training with SSE stream) ---
        async startForge() {
            if (!this.forge.name || !this.forge.topic) {
                alert("Inserisci sia il nome dell'esperimento che il topic didattico.");
                return;
            }

            this.forge.isRunning = true;
            this.forge.progress = '0%';
            this.forge.terminalLogs = ['> Inizializzazione pipeline Forge (Engine: Soup Backend)...'];

            try {
                const res = await fetch('/api/forge/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name: this.forge.name,
                        model: this.forge.model,
                        topic: this.forge.topic,
                        qlora: this.forge.qlora,
                        method: this.forge.method,
                        turn_type: this.forge.turnType || 'single_turn',
                        target_samples: parseInt(this.forge.targetSamples) || 100
                    })
                });

                if (!res.ok) throw new Error("Errore nell'avvio della forgia");

                const reader = res.body.getReader();
                const decoder = new TextDecoder("utf-8");
                let done = false;

                while (!done) {
                    const { value, done: doneReading } = await reader.read();
                    done = doneReading;
                    if (value) {
                        const chunk = decoder.decode(value, { stream: true });
                        const lines = chunk.split('\n');
                        for (let line of lines) {
                            if (line.startsWith('data: ')) {
                                const data = line.substring(6);
                                if (data === '[DONE]') {
                                    this.forge.isRunning = false;
                                    await this.loadExperiments();
                                    break;
                                }
                                if (data.startsWith('PROGRESS:')) {
                                    this.forge.progress = data.split(':')[1].trim() + '%';
                                } else {
                                    this.forge.terminalLogs.push(data);
                                    this.$nextTick(() => {
                                        const term = document.getElementById('forge-terminal');
                                        if (term) term.scrollTop = term.scrollHeight;
                                    });
                                }
                            }
                        }
                    }
                }
            } catch (e) {
                this.forge.terminalLogs.push(`❌ Errore: ${e.message}`);
                this.forge.isRunning = false;
            }
        },

        clearForgeLogs() {
            this.forge.terminalLogs = ['> Terminale resettato. In attesa di avvio pipeline...'];
            this.forge.progress = '0%';
        },

        selectModelForForge(modelId) {
            this.forge.model = modelId;
            this.tab = 'forge';
        },

        // --- Connectors ---
        async loadConnectors() {
            try {
                const res = await fetch('/api/connectors');
                if (!res.ok) return;
                this.connectors = await res.json();
            } catch (e) {
                console.error("Errore caricamento connettori", e);
            }
        },

        async saveConnector() {
            if (!this.newConn.id || !this.newConn.name || !this.newConn.base_url) {
                alert("ID, Nome e Base URL sono obbligatori.");
                return;
            }
            this.isSavingConn = true;
            this.connMsg = '';

            try {
                const res = await fetch('/api/connectors', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(this.newConn)
                });
                if (res.ok) {
                    this.connMsg = "Connettore salvato con successo!";
                    this.newConn = { id: '', name: '', base_url: 'http://127.0.0.1:11434/v1', api_key: '' };
                    await this.loadConnectors();
                } else {
                    const data = await res.json();
                    this.connMsg = "Errore: " + (data.detail || "Impossibile salvare");
                }
            } catch (e) {
                this.connMsg = "Errore di rete";
            } finally {
                this.isSavingConn = false;
            }
        },

        async deleteConnector(id) {
            if (!confirm(`Sei sicuro di voler rimuovere il connettore "${id}"?`)) return;
            try {
                await fetch(`/api/connectors/${id}`, { method: 'DELETE' });
                await this.loadConnectors();
            } catch (e) {
                console.error("Errore cancellazione connettore", e);
            }
        },

        async testConnector(conn) {
            this.testingConnId = conn.id;
            this.connTestResults[conn.id] = 'Verifica in corso...';
            try {
                const res = await fetch('/api/connectors/test', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        id: conn.id,
                        base_url: conn.base_url
                    })
                });
                const data = await res.json();
                this.connTestResults[conn.id] = (data.status === 'ok' ? '✅ ' : '❌ ') + data.message;
            } catch (e) {
                this.connTestResults[conn.id] = '❌ Errore di rete: ' + e.message;
            } finally {
                this.testingConnId = null;
            }
        },

        setConnectorPreset(preset) {
            if (preset === 'ollama') {
                this.newConn = { id: 'ollama_local', name: 'Ollama Locale', base_url: 'http://127.0.0.1:11434/v1', api_key: '' };
            } else if (preset === 'lmstudio') {
                this.newConn = { id: 'lmstudio', name: 'LM Studio Locale', base_url: 'http://127.0.0.1:1234/v1', api_key: '' };
            } else if (preset === 'gemini') {
                this.newConn = { id: 'google', name: 'Google Gemini (OpenAI Std)', base_url: 'https://generativelanguage.googleapis.com/v1beta/openai/', api_key: '' };
            } else if (preset === 'openai') {
                this.newConn = { id: 'openai', name: 'OpenAI Cloud', base_url: 'https://api.openai.com/v1', api_key: '' };
            }
        }
    };
}
