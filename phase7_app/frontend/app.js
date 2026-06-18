document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const codeInput = document.getElementById('code-input');
    const langSelect = document.getElementById('language-select');
    const analyzeBtn = document.getElementById('analyze-btn');
    const clearBtn = document.getElementById('clear-btn');
    const themeToggle = document.getElementById('theme-toggle');
    
    // State Panels
    const emptyState = document.getElementById('empty-state');
    const loadingState = document.getElementById('loading-state');
    const resultsContent = document.getElementById('results-content');
    
    // Result Elements
    const probValue = document.getElementById('prob-value');
    const riskBadge = document.getElementById('risk-badge');
    const gaugeFill = document.getElementById('gauge-fill');
    
    const metricCc = document.getElementById('metric-cc');
    const metricDepth = document.getElementById('metric-depth');
    const metricFuncs = document.getElementById('metric-funcs');
    const metricLines = document.getElementById('metric-lines');
    
    const securityList = document.getElementById('security-list');
    const issuesCount = document.getElementById('issues-count');
    const complexityList = document.getElementById('complexity-list');
    
    // API URL
    const API_BASE_URL = 'http://localhost:8000/api/v1';

    // Theme Toggle
    themeToggle.addEventListener('click', () => {
        const html = document.documentElement;
        const currentTheme = html.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        html.setAttribute('data-theme', newTheme);
        themeToggle.innerHTML = newTheme === 'dark' ? '<i class="fa-solid fa-moon"></i>' : '<i class="fa-solid fa-sun"></i>';
    });

    // Clear Button
    clearBtn.addEventListener('click', () => {
        codeInput.value = '';
        showState('empty');
    });

    // Analyze Button
    analyzeBtn.addEventListener('click', async () => {
        const code = codeInput.value.trim();
        if (!code) {
            alert('Please enter some code to analyze.');
            return;
        }

        showState('loading');
        simulateLoadingSteps();

        try {
            const response = await fetch(`${API_BASE_URL}/analyze/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    code: code,
                    language: langSelect.value
                })
            });

            if (!response.ok) {
                throw new Error(`API Error: ${response.status}`);
            }

            const data = await response.json();
            renderResults(data);
            showState('results');
        } catch (error) {
            console.error('Analysis failed:', error);
            alert('Analysis failed. Make sure the backend server is running on port 8000.');
            showState('empty');
        }
    });

    function showState(state) {
        emptyState.classList.add('hidden');
        loadingState.classList.add('hidden');
        resultsContent.classList.add('hidden');

        if (state === 'empty') emptyState.classList.remove('hidden');
        if (state === 'loading') loadingState.classList.remove('hidden');
        if (state === 'results') resultsContent.classList.remove('hidden');
    }

    function simulateLoadingSteps() {
        const steps = document.querySelectorAll('.step');
        steps.forEach(s => s.classList.remove('active'));
        
        let currentStep = 0;
        steps[currentStep].classList.add('active');
        
        const interval = setInterval(() => {
            currentStep++;
            if (currentStep < steps.length) {
                steps.forEach(s => s.classList.remove('active'));
                steps[currentStep].classList.add('active');
            } else {
                clearInterval(interval);
            }
        }, 800);
    }

    function renderResults(data) {
        // 1. Bug Probability & Risk
        const probPercent = Math.round(data.bug_probability * 100);
        probValue.textContent = `${probPercent}%`;
        
        // Update Gauge
        // Total dash array is ~125.6. Dash offset = 125.6 - (percent * 125.6)
        const offset = 125.6 - (data.bug_probability * 125.6);
        gaugeFill.style.strokeDashoffset = offset;
        
        // Colors based on risk
        let color, badgeClass;
        if (data.risk_level === 'CRITICAL' || data.risk_level === 'HIGH') {
            color = 'var(--danger)';
            badgeClass = 'badge-high';
        } else if (data.risk_level === 'MEDIUM') {
            color = 'var(--warning)';
            badgeClass = 'badge-medium';
        } else {
            color = 'var(--success)';
            badgeClass = 'badge-low';
        }
        
        gaugeFill.style.stroke = color;
        riskBadge.textContent = `${data.risk_level} RISK`;
        riskBadge.className = `badge ${badgeClass}`;

        // 2. Metrics
        metricCc.textContent = data.metrics.cyclomatic_complexity;
        metricDepth.textContent = data.metrics.max_nesting_depth;
        metricFuncs.textContent = data.metrics.function_count;
        metricLines.textContent = data.metrics.total_lines;

        // 3. Security Issues
        securityList.innerHTML = '';
        issuesCount.textContent = data.security_issues.length;
        
        if (data.security_issues.length === 0) {
            securityList.innerHTML = '<div class="issue-item" style="border-left-color: var(--success)"><div class="issue-header"><span class="issue-title"><i class="fa-solid fa-check-circle"></i> No vulnerabilities found</span></div><div class="issue-desc">Your code looks secure based on our static analysis rules.</div></div>';
        } else {
            data.security_issues.forEach(issue => {
                const item = document.createElement('div');
                item.className = `issue-item severity-${issue.severity}`;
                
                let icon = issue.severity === 'HIGH' ? 'fa-triangle-exclamation' : 
                           issue.severity === 'MEDIUM' ? 'fa-circle-exclamation' : 'fa-circle-info';
                           
                item.innerHTML = `
                    <div class="issue-header">
                        <span class="issue-title"><i class="fa-solid ${icon}"></i> ${issue.rule_name}</span>
                        <span class="issue-location">Line ${issue.line_number}</span>
                    </div>
                    <div class="issue-desc">${issue.description}</div>
                    <div class="issue-remedy"><i class="fa-solid fa-wrench"></i> <strong>Fix:</strong> ${issue.remediation}</div>
                `;
                securityList.appendChild(item);
            });
        }

        // 4. Complexity
        complexityList.innerHTML = '';
        if (data.complexity_estimates.length === 0) {
            complexityList.innerHTML = '<div class="complexity-item"><span class="comp-func">N/A</span><span class="comp-class">No functions</span></div>';
        } else {
            data.complexity_estimates.forEach(est => {
                const item = document.createElement('div');
                item.className = 'complexity-item';
                
                let colorClass = '';
                if (est.complexity_class === 'O(1)') colorClass = 'class-o1';
                else if (est.complexity_class === 'O(n)' || est.complexity_class === 'O(log n)') colorClass = 'class-on';
                else if (est.complexity_class.includes('n²') || est.complexity_class.includes('n³')) colorClass = 'class-on2';
                else colorClass = 'class-o2n';
                
                item.innerHTML = `
                    <span class="comp-func">${est.function_name}()</span>
                    <span class="comp-class ${colorClass}">${est.complexity_class}</span>
                    <span style="font-size: 0.8rem; color: var(--text-muted)">Confidence: ${Math.round(est.confidence * 100)}%</span>
                `;
                complexityList.appendChild(item);
            });
        }
    }
});
