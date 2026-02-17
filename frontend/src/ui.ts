import { AgentDTO, LogDTO } from "./connector/dtos";

export default class UI {
    private roundElement: HTMLElement | null;
    private timeElement: HTMLElement | null;
    private agentListElement: HTMLElement | null;

    private selectedAgentName: string | null = null;

    constructor() {
        this.roundElement = document.getElementById("status-round");
        this.timeElement = document.getElementById("status-time");
        this.agentListElement = document.getElementById("agent-list");

        const unfocusBtn = document.getElementById("btn-unfocus");
        if (unfocusBtn) {
            unfocusBtn.onclick = () => {
                this.deselectAgent();
            };
        }

        this.logPanel = document.getElementById("log-panel");
    }

    private logPanel: HTMLElement | null;

    updateStatus(round: number, time: string) {
        if (this.roundElement) this.roundElement.innerText = round.toString();
        if (this.timeElement) this.timeElement.innerText = time;
    }

    updateAgents(agents: AgentDTO[]) {
        if (!this.agentListElement) return;

        // If an agent is selected, update their details live
        if (this.selectedAgentName) {
            const selected = agents.find(a => a.name === this.selectedAgentName);
            if (selected) {
                this.renderAgentDetails(selected);
            }
        }

        this.agentListElement.innerHTML = ""; // Clear list

        agents.forEach(agent => {
            const card = document.createElement("div");
            card.className = "agent-card";
            if (this.selectedAgentName === agent.name) {
                card.style.borderLeftColor = "#ffeb3b"; // Highlight selected in list
                card.style.backgroundColor = "#fff9c4";
            }

            card.onclick = () => {
                this.selectAgent(agent);
            };

            const nameDiv = document.createElement("div");
            nameDiv.className = "agent-name";
            nameDiv.innerText = `${agent.emoji} ${agent.name.replace(/_/g, " ")}`;

            const activityDiv = document.createElement("div");
            activityDiv.className = "agent-activity";
            activityDiv.innerText = agent.activity;

            card.appendChild(nameDiv);
            card.appendChild(activityDiv);
            this.agentListElement?.appendChild(card);
        });
    }

    selectAgent(agent: AgentDTO) {
        if (this.selectedAgentName === agent.name) return;

        this.selectedAgentName = agent.name;
        this.renderAgentDetails(agent);
        this.clearLogs();

        // Dispatch event for GameScene to pick up (highlighting)
        // Corrected event name to match Game.ts listener
        window.dispatchEvent(new CustomEvent('agent-subscribe', { detail: { name: agent.name } }));
    }

    deselectAgent() {
        this.selectedAgentName = null;
        this.clearLogs();
        const detailsEl = document.getElementById("agent-details");
        if (detailsEl) {
            detailsEl.style.display = "none";
        }

        // Dispatch event for GameScene to pick up
        window.dispatchEvent(new CustomEvent('deselect-agent'));
    }

    renderAgentDetails(agent: AgentDTO) {
        const DETAILS_ID = "agent-details";
        const detailsEl = document.getElementById(DETAILS_ID);
        if (!detailsEl) return;

        detailsEl.style.display = "block";

        const setText = (id: string, text: string) => {
            const el = document.getElementById(id);
            if (el) el.innerText = text;
        }

        setText("detail-name", agent.name.replace(/_/g, " "));
        setText("detail-age", agent.age ? agent.age.toString() : "N/A");
        setText("detail-desc", agent.description);
        setText("detail-loc", agent.location);
        setText("detail-act", agent.activity);
        setText("detail-act", agent.activity);
    }

    renderLogMessage(log: LogDTO) {
        if (!this.logPanel) return;

        // Client-side filtering is no longer needed as the backend emits to specific agent rooms
        // if (this.selectedAgentName !== log.agent) return;
        console.log(log)
        // If it's the first log, clear the "placeholder" text
        if (this.logPanel.children.length > 0 && this.logPanel.children[0].tagName === "DIV" && (this.logPanel.children[0] as HTMLElement).innerText.includes("Select an agent")) {
            this.logPanel.innerHTML = "";
        }

        const logItem = document.createElement("div");
        logItem.style.marginBottom = "5px";
        logItem.style.borderBottom = "1px solid #444";
        logItem.style.paddingBottom = "5px";

        const meta = document.createElement("span");
        meta.style.color = "#888";
        meta.style.fontSize = "0.9em";
        meta.innerText = `${log.timestamp.split(' ')[1]} [${log.level}]: `;

        const msg = document.createElement("span");
        if (log.level === "DEBUG") msg.style.color = "#aaa";
        else if (log.level === "WARNING") msg.style.color = "#ffeb3b";
        else if (log.level === "ERROR") msg.style.color = "#f44336";
        else msg.style.color = "#fff";

        msg.innerText = log.message;

        logItem.appendChild(meta);
        logItem.appendChild(msg);
        this.logPanel.appendChild(logItem);

        // Auto scroll
        this.logPanel.scrollTop = this.logPanel.scrollHeight;
    }

    clearLogs() {
        if (this.logPanel) {
            this.logPanel.innerHTML = '<div style="color: #aaa; font-style: italic;">Select an agent to see logs...</div>';
        }
    }
}
