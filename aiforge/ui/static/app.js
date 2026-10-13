document.addEventListener("DOMContentLoaded", () => {
    // Navigation
    const navItems = document.querySelectorAll('.nav-item');
    const sections = document.querySelectorAll('.section');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            navItems.forEach(n => {
                n.classList.remove('active', 'bg-slate-100', 'text-slate-900');
                n.classList.add('text-slate-600');
            });
            sections.forEach(s => s.classList.remove('active'));
            
            item.classList.add('active', 'bg-slate-100', 'text-slate-900');
            item.classList.remove('text-slate-600');
            document.getElementById(item.dataset.target).classList.add('active');
            
            if (item.dataset.target === 'experiments') {
                loadExperiments();
            } else if (item.dataset.target === 'models') {
                loadModels();
            } else if (item.dataset.target === 'connectors') {
                loadConnectors();
            } else if (item.dataset.target === 'merge-lab') {
                loadMergeModels();
            }
        });
    });

    // Status Check
    async function checkStatus() {
        try {
            const res = await fetch('/api/status');
            const data = await res.json();
            
            const expCountEl = document.getElementById('stat-total-experiments') || document.getElementById('dash-exp-count');
            if (expCountEl) expCountEl.textContent = data.experiments_count;
            
            const serverBadge = document.getElementById('server-status');
            
            if (data.llm_server === 'online') {
                serverBadge.innerHTML = '<span class="w-1.5 h-1.5 mr-1.5 rounded-full bg-emerald-500"></span>Online';
                serverBadge.className = "inline-flex items-center px-2 py-0.5 rounded-full font-medium bg-emerald-50 text-emerald-700 border border-emerald-200";
                document.getElementById('btn-start-server').disabled = true;
                document.getElementById('btn-stop-server').disabled = false;
                if(document.getElementById('server-msg').textContent === "Server is starting...") {
                    document.getElementById('server-msg').textContent = "Server is running.";
                }
            } else {
                serverBadge.innerHTML = '<span class="w-1.5 h-1.5 mr-1.5 rounded-full bg-rose-500"></span>Offline';
                serverBadge.className = "inline-flex items-center px-2 py-0.5 rounded-full font-medium bg-rose-50 text-rose-700 border border-rose-200";
                document.getElementById('btn-start-server').disabled = false;
                document.getElementById('btn-stop-server').disabled = true;
            }
        } catch (e) {
            console.error(e);
        }
    }

    // Load Experiments
    async function loadExperiments() {
        try {
            const res = await fetch('/api/experiments');
            const exps = await res.json();
            const tbody = document.getElementById('experiments-table-body') || document.querySelector('#exp-table tbody');
            tbody.innerHTML = '';
            
            const select = document.getElementById('llm-experiment-select');
            const headerSelect = document.getElementById('global-header-model-select');
            
            // Build groups cleanly
            const baseOptionsHtml = `
                <optgroup label="Modelli Base di Partenza (Out-of-the-box)">
                    <option value="BASE:Qwen/Qwen2.5-Coder-0.5B-Instruct">Qwen 2.5 Coder 0.5B (Puro - Senza Adapter)</option>
                    <option value="BASE:Qwen/Qwen2.5-Coder-1.5B-Instruct">Qwen 2.5 Coder 1.5B (Puro - Senza Adapter)</option>
                    <option value="BASE:openai-community/gpt2">GPT-2 Base (Puro)</option>
                </optgroup>
            `;

            let tunedOptionsHtml = '<optgroup label="Modelli Fine-Tuned (I tuoi Adapter)">';
            let hasTuned = false;
            
            exps.forEach(exp => {
                const isLLM = exp.config?.domain === 'llm';
                const expId = exp.id || exp.experiment_id || 'unknown';
                const expName = exp.experiment_name || exp.name || 'Unnamed';
                
                // Add to table
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="px-4 py-3 font-mono text-xs"><code>${expId.substring(0,8)}...</code></td>
                    <td class="px-4 py-3 font-medium text-slate-900">${expName}</td>
                    <td class="px-4 py-3 text-slate-600">${exp.config?.domain || 'N/A'}</td>
                    <td class="px-4 py-3 text-slate-600">${exp.config?.task || 'N/A'}</td>
                    <td class="px-4 py-3 text-slate-500 text-xs">${exp.timestamp ? new Date(exp.timestamp).toLocaleString() : 'N/A'}</td>
                `;
                tr.addEventListener('click', () => showExperimentDetail(expId));
                tbody.appendChild(tr);
                
                // Add to LLM select if it's an LLM
                if (isLLM) {
                    hasTuned = true;
                    tunedOptionsHtml += `<option value="${expId}">${expName} (${exp.timestamp ? new Date(exp.timestamp).toLocaleDateString() : 'N/A'})</option>`;
                }
            });

            if (!hasTuned) {
                tunedOptionsHtml += '<option value="" disabled>Nessun adapter addestrato presente</option>';
            }
            tunedOptionsHtml += '</optgroup>';

            if (select) {
                select.innerHTML = '<option value="">-- Seleziona un modello da avviare --</option>' + tunedOptionsHtml + baseOptionsHtml;
            }
            if (headerSelect) {
                headerSelect.innerHTML = tunedOptionsHtml + baseOptionsHtml;
            }
        } catch (e) {
            console.error(e);
        }
    }

    // Server Controls
    document.getElementById('btn-start-server').addEventListener('click', async () => {
        const expId = document.getElementById('llm-experiment-select').value;
        if (!expId) return alert("Please select an experiment");
        
        try {
            const res = await fetch('/api/server/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ experiment_id: expId })
            });
            if (res.ok) {
                document.getElementById('server-msg').style.color = 'green';
                document.getElementById('server-msg').textContent = "Server is starting...";
                checkStatus();
            } else {
                const data = await res.json();
                alert("Error: " + data.detail);
            }
        } catch(e) {
            alert("Network error");
        }
    });
    
    // Global Header Model Switcher and Toggle
    const headerSelect = document.getElementById('global-header-model-select');
    const headerBtnToggle = document.getElementById('header-btn-toggle-server');
    const dashSelect = document.getElementById('llm-experiment-select');

    if (headerSelect && dashSelect) {
        headerSelect.addEventListener('change', () => {
            dashSelect.value = headerSelect.value;
        });
        dashSelect.addEventListener('change', () => {
            headerSelect.value = dashSelect.value;
        });
    }

    if (headerBtnToggle) {
        headerBtnToggle.addEventListener('click', async () => {
            const serverBadge = document.getElementById('server-status');
            const isOnline = serverBadge.textContent.includes('Online');
            if (isOnline) {
                await fetch('/api/server/stop', { method: 'POST' });
                checkStatus();
            } else {
                const modelVal = (headerSelect ? headerSelect.value : '') || (dashSelect ? dashSelect.value : '');
                if (!modelVal) return alert("Seleziona un modello da avviare");
                await fetch('/api/server/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ experiment_id: modelVal })
                });
                checkStatus();
            }
        });
    }

    // Show Experiment Detail
    async function showExperimentDetail(expId) {
        try {
            const res = await fetch(`/api/experiments/${expId}`);
            const data = await res.json();
            
            document.getElementById('exp-list-container').style.display = 'none';
            document.getElementById('exp-detail-container').style.display = 'block';
            
            document.getElementById('detail-name').textContent = data.name || expId;
            document.getElementById('detail-json').textContent = JSON.stringify(data, null, 2);
        } catch (e) {
            console.error(e);
        }
    }

    document.getElementById('btn-back-exps').addEventListener('click', () => {
        document.getElementById('exp-list-container').style.display = 'block';
        document.getElementById('exp-detail-container').style.display = 'none';
    });

    // LLM Playground Chat Interaction (ChatGPT Style)
    const promptInput = document.getElementById('prompt') || document.getElementById('llm-prompt');
    const chatContainer = document.getElementById('chat-messages-container');
    const latencyEl = document.getElementById('chat-latency');
    const errBox = document.getElementById('llm-error');
    const btnGen = document.getElementById('btn-generate');

    function appendMessage(sender, text) {
        if (!chatContainer) return;
        const isUser = sender === 'user';
        const msgWrapper = document.createElement('div');
        msgWrapper.className = `flex items-start space-x-3 max-w-3xl ${isUser ? 'ml-auto flex-row-reverse space-x-reverse' : ''}`;
        
        const avatar = document.createElement('div');
        avatar.className = `w-7 h-7 rounded-full flex items-center justify-center text-xs font-mono font-bold shrink-0 mt-0.5 ${isUser ? 'bg-slate-200 text-slate-700' : 'bg-slate-900 text-white'}`;
        avatar.textContent = isUser ? 'TU' : 'AI';

        const bubble = document.createElement('div');
        bubble.className = `rounded-2xl px-4 py-3 text-sm leading-relaxed max-w-2xl whitespace-pre-wrap ${isUser ? 'bg-slate-900 text-white rounded-tr-sm shadow-sm' : 'bg-white border border-slate-200 text-slate-800 rounded-tl-sm shadow-sm'}`;
        bubble.textContent = text;

        msgWrapper.appendChild(avatar);
        msgWrapper.appendChild(bubble);
        chatContainer.appendChild(msgWrapper);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        return bubble;
    }

    async function handleChatSubmit() {
        const text = promptInput.value.trim();
        if (!text) return;
        
        const maxTokens = parseInt(document.getElementById('max-tokens').value) || 120;
        const temp = parseFloat(document.getElementById('temperature').value) || 0.3;

        // Append user bubble
        appendMessage('user', text);
        promptInput.value = '';
        promptInput.style.height = 'auto';

        if (errBox) errBox.classList.add('hidden');
        btnGen.disabled = true;

        // Loading bubble
        const aiBubble = appendMessage('ai', 'Elaborazione in corso sulla GPU Apple Silicon...');

        try {
            const res = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: text,
                    max_new_tokens: maxTokens,
                    temperature: temp
                })
            });

            const data = await res.json();
            if (res.ok) {
                aiBubble.textContent = data.text;
                if (latencyEl) latencyEl.textContent = `${data.inference_ms.toFixed(0)} ms`;
            } else {
                aiBubble.textContent = `❌ ${data.detail || 'Errore nella generazione.'}`;
                if (errBox) {
                    errBox.textContent = data.detail;
                    errBox.classList.remove('hidden');
                }
            }
        } catch(e) {
            aiBubble.textContent = '❌ Errore di rete: assicurati che il server LLM sia avviato dalla Dashboard.';
        } finally {
            btnGen.disabled = false;
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }

    btnGen.addEventListener('click', handleChatSubmit);

    // Support Enter to send (Shift+Enter for newline)
    if (promptInput) {
        promptInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleChatSubmit();
            }
        });
    }

    const btnClearChat = document.getElementById('btn-clear-chat');
    if (btnClearChat) {
        btnClearChat.addEventListener('click', () => {
            if (chatContainer) {
                chatContainer.innerHTML = `
                    <div class="flex items-start space-x-3 max-w-3xl">
                        <div class="w-7 h-7 rounded-full bg-slate-900 text-white flex items-center justify-center text-xs font-mono font-bold shrink-0 mt-0.5">AI</div>
                        <div class="bg-white border border-slate-200 rounded-2xl rounded-tl-sm px-4 py-3 text-sm text-slate-800 shadow-sm leading-relaxed max-w-2xl">
                            Chat azzerata. Chiedimi qualsiasi compito in linguaggio C o altro algoritmo!
                        </div>
                    </div>
                `;
            }
        });
    }

    

    // --- Forge UI Logic ---
    const btnForgeStart = document.getElementById('btn-forge-start');
    if (btnForgeStart) {
        btnForgeStart.addEventListener('click', async () => {
            const name = document.getElementById('forge-name').value.trim();
            const model = document.getElementById('forge-model').value;
            const topic = document.getElementById('forge-topic').value.trim();
            const term = document.getElementById('forge-terminal');
            const progress = document.getElementById('forge-progress-bar');
            const progressTxt = document.getElementById('forge-progress-text');

            if (!name || !topic) {
                alert("Compila il nome dell'esperimento e il topic!");
                return;
            }

            btnForgeStart.disabled = true;
            btnForgeStart.textContent = "Forgia in corso...";
            progress.style.width = '0%';
            progressTxt.textContent = '0%';
            term.innerHTML = '';
            
            try {
                const res = await fetch('/api/forge/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name, model, topic })
                });

                if (!res.ok) {
                    throw new Error("Errore nell'avvio della pipeline");
                }

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
                                    btnForgeStart.disabled = false;
                                    btnForgeStart.textContent = "Genera Dati & Addestra (Forgia)";
                                    loadExperiments(); // Refresh list
                                    break;
                                }
                                
                                if (data.startsWith('PROGRESS:')) {
                                    const pct = data.split(':')[1].trim() + '%';
                                    progress.style.width = pct;
                                    progressTxt.textContent = pct;
                                } else {
                                    const div = document.createElement('div');
                                    div.textContent = data;
                                    term.appendChild(div);
                                    term.scrollTop = term.scrollHeight;
                                }
                            }
                        }
                    }
                }
            } catch (e) {
                const div = document.createElement('div');
                div.style.color = '#ff6b6b';
                div.textContent = "Errore di connessione: " + e.message;
                term.appendChild(div);
                btnForgeStart.disabled = false;
                btnForgeStart.textContent = "Genera Dati & Addestra (Forgia)";
            }
        });
    }

    // Connectors Management
    async function loadConnectors() {
        try {
            const res = await fetch('/api/connectors');
            const data = await res.json();
            
            const listDiv = document.getElementById('connectors-list');
            if(!listDiv) return;
            listDiv.innerHTML = '';
            
            data.forEach(conn => {
                const card = document.createElement('div');
                card.className = 'border border-slate-200 rounded-lg p-4 bg-white flex justify-between items-center';
                
                const keyStatus = conn.has_key ? 
                    '<span class="text-[10px] uppercase font-bold tracking-wider text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">Key Presente</span>' : 
                    '<span class="text-[10px] uppercase font-bold tracking-wider text-slate-400 bg-slate-50 px-2 py-0.5 rounded border border-slate-200">Key Vuota</span>';
                
                card.innerHTML = `
                    <div>
                        <div class="flex items-center space-x-2">
                            <h4 class="text-sm font-semibold text-slate-900">${conn.name}</h4>
                            ${keyStatus}
                        </div>
                        <div class="text-xs text-slate-500 mt-1 font-mono">${conn.base_url || 'N/A'}</div>
                    </div>
                    <button class="text-xs text-rose-500 hover:text-rose-700 font-medium" onclick="deleteConnector('${conn.id}')">Rimuovi</button>
                `;
                listDiv.appendChild(card);
            });
        } catch (e) {
            console.error("Failed to load connectors", e);
        }
    }

    window.deleteConnector = async function(id) {
        if(!confirm('Sicuro di voler rimuovere questo connettore?')) return;
        await fetch(`/api/connectors/${id}`, { method: 'DELETE' });
        loadConnectors();
    loadModels();
    };

    const btnSaveConn = document.getElementById('btn-save-connector');
    if (btnSaveConn) {
        btnSaveConn.addEventListener('click', async () => {
            const id = document.getElementById('conn-id').value.trim();
            const name = document.getElementById('conn-name').value.trim();
            const base_url = document.getElementById('conn-url').value.trim();
            const api_key = document.getElementById('conn-key').value;
            
            if(!id || !name || !base_url) return alert("ID, Nome e Base URL sono obbligatori!");
            
            btnSaveConn.disabled = true;
            const msg = document.getElementById('connector-msg');
            
            try {
                const res = await fetch('/api/connectors', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ id, name, base_url, api_key })
                });
                
                if (res.ok) {
                    msg.style.color = 'green';
                    msg.textContent = 'Connettore salvato!';
                    document.getElementById('conn-id').value = '';
                    document.getElementById('conn-name').value = '';
                    document.getElementById('conn-url').value = '';
                    document.getElementById('conn-key').value = '';
                    loadConnectors();
    loadModels();
                } else {
                    msg.style.color = 'red';
                    msg.textContent = 'Errore nel salvataggio.';
                }
            } catch(e) {
                msg.style.color = 'red';
                msg.textContent = 'Errore di rete.';
            }
            btnSaveConn.disabled = false;
            setTimeout(() => { msg.textContent = ''; }, 3000);
        });
    }


    // Initial check
    checkStatus();
    loadExperiments();
    loadConnectors();
    loadModels();
    setInterval(checkStatus, 5000);
});

// Models Management
    async function loadModels() {
        try {
            const res = await fetch('/api/models');
            const models = await res.json();
            
            const listDiv = document.getElementById('models-list-container');
            if(!listDiv) return;
            listDiv.innerHTML = '';
            
            models.forEach(m => {
                const card = document.createElement('div');
                card.className = `border border-slate-200 rounded-lg p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4 transition-colors ${m.downloaded ? 'opacity-75' : 'hover:border-slate-300'}`;
                
                const badge = m.recommended ? 
                    '<span class="bg-emerald-50 text-emerald-700 text-xs px-2 py-0.5 rounded font-medium border border-emerald-200">Consigliato M1</span>' : '';
                
                const button = m.downloaded ? 
                    `<span class="text-xs text-slate-400 font-medium px-3 py-1 bg-slate-50 rounded border border-slate-200">Già scaricato in Cache</span>` :
                    `<button onclick="downloadModel('${m.id}')" class="bg-slate-900 hover:bg-slate-800 text-white text-xs font-medium px-4 py-2.5 rounded-lg shrink-0 transition-colors">Scarica per Uso Locale</button>`;
                
                card.innerHTML = `
                    <div>
                        <div class="flex items-center space-x-2">
                            <h3 class="font-semibold text-slate-900">${m.name}</h3>
                            ${badge}
                        </div>
                        <p class="text-xs text-slate-500 mt-1">${m.desc || ''}</p>
                        <span class="text-xs font-mono text-slate-400 mt-2 block">HuggingFace ID: ${m.id}</span>
                    </div>
                    ${button}
                `;
                listDiv.appendChild(card);
            });
            
            // Re-bind global download model function if needed, but it's defined in index.html script block!
            // Wait, we need to update the global downloadModel to refresh list on success.
        } catch (e) {
            console.error("Failed to load models", e);
        }
    }
    
    // override global downloadModel to refresh after success
    window.downloadModel = async function(modelId) {
        const statusBox = document.getElementById('model-download-status');
        statusBox.classList.remove('hidden');
        statusBox.textContent = `Download del modello ${modelId} in corso da Hugging Face... Attendere (può impiegare minuti).`;
        try {
            const res = await fetch('/api/models/download', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_id: modelId })
            });
            const data = await res.json();
            if (res.ok) {
                statusBox.textContent = `✅ ${data.message}`;
                loadModels(); // Refresh list to show "Già scaricato"
            } else {
                statusBox.textContent = `❌ ${data.detail}`;
            }
        } catch(e) {
            statusBox.textContent = "Errore di rete durante il download.";
        }
    };


// ==================== MERGE LAB ====================
async function loadMergeModels() {
    const selectA = document.getElementById('merge-model-a');
    const selectB = document.getElementById('merge-model-b');
    if (!selectA || !selectB) return;
    
    try {
        const [modelsRes, expsRes] = await Promise.all([
            fetch('/api/models'),
            fetch('/api/experiments')
        ]);
        const baseModels = await modelsRes.json();
        const experiments = await expsRes.json();
        
        let options = '<option value="">-- Seleziona --</option>';
        baseModels.forEach(m => {
            options += `<option value="BASE:${m.id}">[Base LLM] ${m.name}</option>`;
        });
        experiments.forEach(e => {
            options += `<option value="${e.id}">[Artifact] ${e.experiment_name} (${e.id})</option>`;
        });
        
        selectA.innerHTML = options;
        selectB.innerHTML = options;
    } catch (e) {
        console.error("Failed to load models for merge", e);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    // Refresh models when merge lab is clicked
    const mergeNav = document.querySelector('[data-target="merge-lab"]');
    if (mergeNav) {
        mergeNav.addEventListener('click', loadMergeModels);
    }
    
    const btnPreflight = document.getElementById('btn-preflight');
    const btnExecute = document.getElementById('btn-execute-merge');
    const resultsDiv = document.getElementById('merge-preflight-results');
    const alphaSlider = document.getElementById('merge-alpha');
    const alphaValSpan = document.getElementById('alpha-val');
    const postActionsDiv = document.getElementById('merge-post-actions');
    const btnRunEval = document.getElementById('btn-run-eval');
    const btnExportOllama = document.getElementById('btn-export-ollama');
    const evalStatusBox = document.getElementById('eval-status-box');

    let lastMergedId = null;
    let lastModelA = null;
    let lastModelB = null;

    if (alphaSlider && alphaValSpan) {
        alphaSlider.addEventListener('input', () => {
            alphaValSpan.textContent = parseFloat(alphaSlider.value).toFixed(2);
        });
    }
    
    if (btnPreflight) {
        btnPreflight.addEventListener('click', async () => {
            const modelA = document.getElementById('merge-model-a').value;
            const modelB = document.getElementById('merge-model-b').value;
            const strategy = document.getElementById('merge-strategy').value;
            
            if (!modelA || !modelB) {
                alert("Seleziona entrambi i modelli.");
                return;
            }
            
            btnPreflight.textContent = "Verifica in corso...";
            resultsDiv.classList.remove('hidden');
            resultsDiv.innerHTML = "Controllo requisiti hardware e semantica...";
            
            try {
                const res = await fetch(`/api/merge/preflight?model_a=${encodeURIComponent(modelA)}&model_b=${encodeURIComponent(modelB)}&strategy=${encodeURIComponent(strategy)}`);
                const data = await res.json();
                
                let html = `<strong>Stato:</strong> ${data.status}<br>`;
                html += `<strong>RAM Stimata:</strong> ~${data.estimated_ram_mb} MB<br>`;
                
                if (data.warnings && data.warnings.length > 0) {
                    html += `<div class="mt-2 text-amber-600"><strong>Warnings:</strong><ul>`;
                    data.warnings.forEach(w => html += `<li>- ${w}</li>`);
                    html += `</ul></div>`;
                }
                
                if (data.incompatibilities && data.incompatibilities.length > 0) {
                    html += `<div class="mt-2 text-red-600"><strong>Incompatibilità:</strong><ul>`;
                    data.incompatibilities.forEach(i => html += `<li>- ${i}</li>`);
                    html += `</ul></div>`;
                    btnExecute.disabled = true;
                } else {
                    btnExecute.disabled = false;
                }
                
                resultsDiv.innerHTML = html;
            } catch (e) {
                resultsDiv.innerHTML = `<span class="text-red-600">Errore di connessione: ${e.message}</span>`;
            } finally {
                btnPreflight.textContent = "Verifica Compatibilità";
            }
        });
    }
    
    if (btnExecute) {
        btnExecute.addEventListener('click', async () => {
            const modelA = document.getElementById('merge-model-a').value;
            const modelB = document.getElementById('merge-model-b').value;
            const strategy = document.getElementById('merge-strategy').value;
            const alpha = alphaSlider ? parseFloat(alphaSlider.value) : 0.5;
            
            btnExecute.disabled = true;
            btnExecute.textContent = "Esecuzione in corso...";
            resultsDiv.innerHTML += `<br><span class="text-blue-600">Merge avviato... attendere, l'operazione può richiedere diversi minuti.</span>`;
            if (postActionsDiv) postActionsDiv.classList.add('hidden');
            
            try {
                const res = await fetch('/api/merge/execute', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model_a_id: modelA,
                        model_b_id: modelB,
                        strategy: strategy,
                        alpha: alpha,
                        output_name: `Merged_${strategy}_${Date.now().toString().slice(-4)}`
                    })
                });
                
                const data = await res.json();
                if (res.ok) {
                    lastMergedId = data.experiment_id;
                    lastModelA = modelA;
                    lastModelB = modelB;
                    resultsDiv.innerHTML += `<br><span class="text-green-600 font-bold">Merge completato con successo! ID: ${data.experiment_id}</span>`;
                    if (postActionsDiv) postActionsDiv.classList.remove('hidden');
                    loadExperiments();
                } else {
                    resultsDiv.innerHTML += `<br><span class="text-red-600 font-bold">Errore: ${data.detail}</span>`;
                }
            } catch (e) {
                resultsDiv.innerHTML += `<br><span class="text-red-600 font-bold">Errore di rete: ${e.message}</span>`;
            } finally {
                btnExecute.textContent = "Esegui Merge";
            }
        });
    }

    if (btnRunEval) {
        btnRunEval.addEventListener('click', async () => {
            if (!lastMergedId) return;
            evalStatusBox.classList.remove('hidden');
            evalStatusBox.textContent = "Valutazione numerica in corso su validation slice...";
            btnRunEval.disabled = true;

            try {
                const res = await fetch('/api/merge/evaluate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model_a_id: lastModelA,
                        model_b_id: lastModelB,
                        merged_model_id: lastMergedId
                    })
                });
                const data = await res.json();
                if (res.ok) {
                    const badgeClass = data.release_gate === 'PROMOTED' ? 'text-green-700 font-bold' : 'text-amber-700 font-bold';
                    let output = `[Release Gate]: <span class="${badgeClass}">${data.release_gate}</span>\n`;
                    if (data.metrics && data.metrics.Merged) {
                        const m = data.metrics.Merged;
                        if (m.perplexity) output += `Perplexity Fuso: ${m.perplexity.toFixed(2)}\n`;
                        if (m.accuracy !== undefined) output += `Accuratezza Fuso: ${(m.accuracy * 100).toFixed(1)}%\n`;
                    }
                    evalStatusBox.innerHTML = output;
                } else {
                    evalStatusBox.textContent = `Errore valutazione: ${data.detail || data.error}`;
                }
            } catch (e) {
                evalStatusBox.textContent = `Errore di rete: ${e.message}`;
            } finally {
                btnRunEval.disabled = false;
            }
        });
    }

    if (btnExportOllama) {
        btnExportOllama.addEventListener('click', async () => {
            if (!lastMergedId) return;
            evalStatusBox.classList.remove('hidden');
            evalStatusBox.textContent = "Creazione Modelfile per Ollama in corso...";
            btnExportOllama.disabled = true;

            try {
                const res = await fetch('/api/export/ollama', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ experiment_id: lastMergedId })
                });
                const data = await res.json();
                if (res.ok) {
                    evalStatusBox.innerHTML = `✅ Modelfile creato in: <code>${data.modelfile_path}</code><br>Per registrarlo in Ollama, esegui:<br><pre class="mt-1 p-2 bg-slate-900 text-slate-100 rounded">${data.create_command}</pre>`;
                } else {
                    evalStatusBox.textContent = `Errore export: ${data.detail}`;
                }
            } catch (e) {
                evalStatusBox.textContent = `Errore di rete: ${e.message}`;
            } finally {
                btnExportOllama.disabled = false;
            }
        });
    }
});

