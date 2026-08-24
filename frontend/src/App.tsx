import { FormEvent, useCallback, useEffect, useRef, useState } from 'react'
import {
  Activity, Archive, ArrowUp, BarChart3, Bell, Bot, Brain, Check, CheckCircle2, ChevronRight,
  CircleAlert, CircleStop, Database, FileText, Gauge, KeyRound, Layers3, LayoutDashboard, LogOut,
  Menu, Network, Plus, RefreshCw, Search, Settings, ShieldCheck, Sparkles, Trash2,
  Upload, Users, Wrench, X, XCircle, Zap,
} from 'lucide-react'
import { api, streamRunEvents } from './api'
import type { Analytics, Approval, Document, MemoryRecord, Run, Session, Tool } from './types'

type View = 'mission' | 'runs' | 'knowledge' | 'memory' | 'toolbox' | 'approvals' | 'insights' | 'settings'

const navigation: { id: View; label: string; icon: typeof Activity; detail: string; tone: string }[] = [
  { id: 'mission', label: 'Home', icon: LayoutDashboard, detail: 'Start and guide a goal', tone: 'cyan' },
  { id: 'runs', label: 'Activity', icon: Network, detail: 'Follow every execution', tone: 'blue' },
  { id: 'knowledge', label: 'Library', icon: Database, detail: 'Ground work in sources', tone: 'amber' },
  { id: 'memory', label: 'Memory', icon: Brain, detail: 'Carry useful context', tone: 'violet' },
  { id: 'toolbox', label: 'Capabilities', icon: Wrench, detail: 'Control available tools', tone: 'teal' },
  { id: 'approvals', label: 'Reviews', icon: ShieldCheck, detail: 'Approve sensitive actions', tone: 'coral' },
  { id: 'insights', label: 'Patterns', icon: BarChart3, detail: 'Understand performance', tone: 'lime' },
  { id: 'settings', label: 'Settings', icon: Settings, detail: 'Shape your workspace', tone: 'slate' },
]

const emptyAnalytics: Analytics = {
  total_runs: 0, completed_runs: 0, failed_runs: 0, active_runs: 0, approval_rate: 0,
  average_execution_seconds: 0, p95_execution_seconds: 0, total_llm_calls: 0,
  total_tokens: 0, total_tool_calls: 0, status_counts: {}, daily_runs: [],
}

export default function App() {
  const [session, setSession] = useState<Session | null>(() => {
    try { return JSON.parse(localStorage.getItem('agent-platform-session') ?? 'null') as Session | null } catch { return null }
  })
  const [view, setView] = useState<View>('mission')
  const [mobileNav, setMobileNav] = useState(false)
  const [deckOpen, setDeckOpen] = useState(false)
  const [runs, setRuns] = useState<Run[]>([])
  const [tools, setTools] = useState<Tool[]>([])
  const [memories, setMemories] = useState<MemoryRecord[]>([])
  const [documents, setDocuments] = useState<Document[]>([])
  const [approvals, setApprovals] = useState<Approval[]>([])
  const [analytics, setAnalytics] = useState<Analytics>(emptyAnalytics)
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null)
  const [selectedRun, setSelectedRun] = useState<Run | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const loadData = useCallback(async (quiet = false) => {
    if (!session) return
    if (!quiet) setLoading(true)
    try {
      const workspace = session.workspace.id
      const [nextRuns, nextTools, nextMemories, nextDocuments, nextApprovals, nextAnalytics] = await Promise.all([
        api.runs(session.access_token, workspace), api.tools(session.access_token, workspace),
        api.memories(session.access_token, workspace), api.documents(session.access_token, workspace),
        api.approvals(session.access_token, workspace), api.analytics(session.access_token, workspace),
      ])
      setRuns(nextRuns); setTools(nextTools); setMemories(nextMemories); setDocuments(nextDocuments)
      setApprovals(nextApprovals); setAnalytics(nextAnalytics); setError('')
      if (selectedRunId) setSelectedRun(await api.run(session.access_token, selectedRunId))
    } catch (err) { setError(message(err)) } finally { if (!quiet) setLoading(false) }
  }, [session, selectedRunId])

  useEffect(() => { void loadData() }, [loadData])
  useEffect(() => {
    function handleShortcut(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null
      const editing = target?.tagName === 'INPUT' || target?.tagName === 'TEXTAREA' || target?.tagName === 'SELECT'
      if (event.key === 'Escape') { setDeckOpen(false); setMobileNav(false) }
      if (event.key === '/' && !editing) { event.preventDefault(); setDeckOpen(current => !current) }
      if (event.altKey && /^[1-8]$/.test(event.key)) {
        event.preventDefault(); setView(navigation[Number(event.key) - 1].id); setDeckOpen(false)
      }
    }
    window.addEventListener('keydown', handleShortcut)
    return () => window.removeEventListener('keydown', handleShortcut)
  }, [])
  useEffect(() => {
    if (!session || !runs.some(run => ['queued', 'running'].includes(run.status))) return
    const timer = window.setInterval(() => void loadData(true), 2500)
    return () => window.clearInterval(timer)
  }, [session, runs, loadData])

  function completeAuth(next: Session) {
    localStorage.setItem('agent-platform-session', JSON.stringify(next))
    setSession(next)
  }
  function logout() { localStorage.removeItem('agent-platform-session'); setSession(null) }

  if (!session) return <AuthScreen onSuccess={completeAuth} />
  const pendingApprovals = approvals.filter(item => item.status === 'pending').length

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar-inner">
          <button className="brand" onClick={() => { setView('mission'); setDeckOpen(false) }}><span className="brand-mark"><Sparkles size={18} /></span><span><strong>Learning BOT</strong><small>agent workspace</small></span></button>
          <nav className="desktop-nav" aria-label="Workspace sections">
            {navigation.map(item => <button key={item.id} className={`${view === item.id ? 'active' : ''} tone-${item.tone}`} onClick={() => { setView(item.id); setDeckOpen(false) }}><item.icon size={16} /><span>{item.label}</span>{item.id === 'approvals' && pendingApprovals > 0 && <b>{pendingApprovals}</b>}</button>)}
          </nav>
          <div className="top-actions">
            <button className={`deck-trigger ${deckOpen ? 'active' : ''}`} title="Open workspace map" onClick={() => setDeckOpen(current => !current)}><Search size={16} /><span>Navigate</span><kbd>/</kbd></button>
            <button className="icon-button" title="Refresh workspace" onClick={() => void loadData()}><RefreshCw size={17} className={loading ? 'spinning' : ''} /></button>
            <button className="icon-button" title="Items that need your attention"><Bell size={17} />{pendingApprovals > 0 && <i />}</button>
            <button className="profile-orb" title="Open settings" onClick={() => setView('settings')}>{initials(session.user.display_name)}</button>
            <button className="icon-button mobile-menu" title="Open navigation" onClick={() => setMobileNav(current => !current)}><Menu size={19} /></button>
          </div>
        </div>
      </header>
      {deckOpen && <WorkspaceDeck active={view} approvals={pendingApprovals} analytics={analytics} onSelect={item => { setView(item); setDeckOpen(false) }} onClose={() => setDeckOpen(false)} />}
      {mobileNav && <div className="mobile-drawer"><div><span>Move through the workspace</span><button className="icon-button" onClick={() => setMobileNav(false)}><X size={18} /></button></div>{navigation.map(item => <button key={item.id} className={view === item.id ? 'active' : ''} onClick={() => { setView(item.id); setMobileNav(false) }}><span className={`nav-symbol tone-${item.tone}`}><item.icon size={17} /></span><span><strong>{item.label}</strong><small>{item.detail}</small></span>{item.id === 'approvals' && pendingApprovals > 0 && <b>{pendingApprovals}</b>}<ChevronRight size={15} /></button>)}</div>}
      {mobileNav && <button aria-label="Close navigation" className="nav-scrim" onClick={() => setMobileNav(false)} />}
      <main>
        {error && <div className="global-error"><CircleAlert size={17} /><span>{error}</span><button aria-label="Dismiss error" onClick={() => setError('')}><X size={16} /></button></div>}
        <div className={`workspace view-${view}`}>
          <div className="view-stage" key={view}>
          {view === 'mission' && <MissionView session={session} runs={runs} analytics={analytics} onSelect={selectRun} onCreated={watchRun} onNavigate={setView} />}
          {view === 'runs' && <RunsView session={session} runs={runs} selected={selectedRun} onSelect={selectRun} onRefresh={() => void loadData()} />}
          {view === 'knowledge' && <KnowledgeView session={session} documents={documents} refresh={loadData} />}
          {view === 'memory' && <MemoryView session={session} memories={memories} refresh={loadData} />}
          {view === 'toolbox' && <ToolboxView session={session} tools={tools} refresh={loadData} />}
          {view === 'approvals' && <ApprovalsView session={session} approvals={approvals} refresh={loadData} />}
          {view === 'insights' && <InsightsView analytics={analytics} runs={runs} />}
          {view === 'settings' && <SettingsView session={session} onLogout={logout} />}
          </div>
        </div>
      </main>
    </div>
  )

  async function selectRun(id: string) {
    setSelectedRunId(id); setSelectedRun(await api.run(session!.access_token, id))
  }
  async function watchRun(run: Run) {
    setSelectedRunId(run.id); setSelectedRun(run); setView('runs'); await loadData(true)
    const controller = new AbortController()
    void streamRunEvents(session!.access_token, run.id, () => void loadData(true), controller.signal).catch(() => undefined)
  }
}

function WorkspaceDeck({ active, approvals, analytics, onSelect, onClose }: { active: View; approvals: number; analytics: Analytics; onSelect: (view: View) => void; onClose: () => void }) {
  return <><button className="deck-scrim" aria-label="Close workspace map" onClick={onClose} /><section className="workspace-deck"><div className="deck-content"><div className="deck-summary"><p className="eyebrow">Workspace map</p><h2>Move between every layer of your agent system.</h2><p>Each area exposes a different part of the same durable workflow, without losing the current run or context.</p><div className="deck-signals"><span><i />{analytics.active_runs} active</span><span>{analytics.total_runs} total runs</span><span>{approvals} awaiting review</span></div></div><div className="deck-grid">{navigation.map((item, index) => <button key={item.id} className={`${active === item.id ? 'active' : ''} tone-${item.tone}`} onClick={() => onSelect(item.id)}><span className="nav-symbol"><item.icon size={18} /></span><span><strong>{item.label}</strong><small>{item.detail}</small></span><kbd>Alt {index + 1}</kbd></button>)}</div></div></section></>
}

function AuthScreen({ onSuccess }: { onSuccess: (session: Session) => void }) {
  const [registering, setRegistering] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setBusy(true); setError('')
    const data = new FormData(event.currentTarget)
    try {
      const session = registering
        ? await api.register(String(data.get('email')), String(data.get('password')), String(data.get('name')))
        : await api.login(String(data.get('email')), String(data.get('password')))
      onSuccess(session)
    } catch (err) { setError(message(err)) } finally { setBusy(false) }
  }
  return <div className="auth-page">
    <section className="auth-context">
      <div className="auth-brand"><Sparkles size={21} /> Learning BOT</div>
      <div className="system-map" aria-hidden="true">
        <div className="map-node primary"><Bot size={25} /><span>Manager</span></div>
        <div className="map-line line-a" /><div className="map-line line-b" /><div className="map-line line-c" />
        <div className="map-node node-a"><Search size={20} /><span>Research</span></div>
        <div className="map-node node-b"><FileText size={20} /><span>Write</span></div>
        <div className="map-node node-c"><ShieldCheck size={20} /><span>Review</span></div>
      </div>
      <div><p className="eyebrow">A calmer way to work with AI</p><h1>Bring an idea.<br />Build clarity together.</h1><p>Your agent team can research, write, and review while keeping every important choice visible and in your hands.</p></div>
      <div className="auth-signals"><span><CheckCircle2 size={16} /> Work that stays with you</span><span><ShieldCheck size={16} /> You approve sensitive actions</span><span><Activity size={16} /> Progress you can follow</span></div>
    </section>
    <section className="auth-form-wrap"><form className="auth-form" onSubmit={submit}>
      <div className="auth-heading"><p className="eyebrow">{registering ? 'Begin your workspace' : 'Welcome back'}</p><h2>{registering ? 'Create your account' : 'Sign in to continue'}</h2></div>
      {registering && <label>Display name<input name="name" required minLength={2} placeholder="Your name" /></label>}
      <label>Email address<input name="email" type="email" required placeholder="you@example.com" /></label>
      <label>Password<input name="password" type="password" required minLength={10} placeholder="At least 10 characters" /></label>
      {error && <div className="form-error"><CircleAlert size={16} />{error}</div>}
      <button className="primary-button auth-submit" disabled={busy}>{busy ? <RefreshCw size={18} className="spinning" /> : <KeyRound size={18} />}{registering ? 'Create workspace' : 'Sign in'}</button>
      <button type="button" className="text-button" onClick={() => { setRegistering(!registering); setError('') }}>{registering ? 'Already have an account? Sign in' : 'New here? Create an account'}</button>
    </form></section>
  </div>
}

function MissionView({ session, runs, analytics, onSelect, onCreated, onNavigate }: { session: Session; runs: Run[]; analytics: Analytics; onSelect: (id: string) => void; onCreated: (run: Run) => void; onNavigate: (view: View) => void }) {
  const [goal, setGoal] = useState('')
  const [mode, setMode] = useState('auto')
  const [busy, setBusy] = useState(false)
  async function launch(event: FormEvent) {
    event.preventDefault(); if (!goal.trim()) return; setBusy(true)
    try { const run = await api.createRun(session.access_token, session.workspace.id, goal, mode); setGoal(''); onCreated(run) } finally { setBusy(false) }
  }
  const suggestions = [
    'Compare a technical decision with clear tradeoffs',
    'Turn an early product idea into a practical plan',
    'Review a proposal for risks and missing assumptions',
  ]
  return <div className="mission-experience">
    <div className="ambient-capabilities" aria-hidden="true">
      <span className="float-cap cap-plan"><Network size={20} /><small>Plan</small></span>
      <span className="float-cap cap-recall"><Brain size={20} /><small>Recall</small></span>
      <span className="float-cap cap-ground"><Database size={20} /><small>Ground</small></span>
      <span className="float-cap cap-act"><Wrench size={20} /><small>Act</small></span>
      <span className="float-cap cap-review"><ShieldCheck size={20} /><small>Review</small></span>
      <span className="float-cap cap-trace"><Activity size={20} /><small>Trace</small></span>
    </div>
    <section className="mission-hero">
      <div className="mission-intro">
        <div className="agent-emblem"><Layers3 size={24} /><span /></div>
        <div className="core-status"><i /><b>Stage 11 is ready</b><span>·</span>one goal, coordinated execution</div>
      </div>
      <form className="ai-composer" onSubmit={launch}>
        <Search size={20} className="composer-search" />
        <textarea value={goal} onChange={e => setGoal(e.target.value)} maxLength={12000} placeholder="Give your agent team a goal..." aria-label="Describe your goal" />
        <div className="composer-footer">
          <div className="composer-meta"><span>{goal.length.toLocaleString()} / 12k</span><button className="send-button" title="Start this work" disabled={busy || goal.trim().length < 3}>{busy ? <RefreshCw size={17} className="spinning" /> : <><Zap size={15} /><b>Start</b></>}</button></div>
        </div>
      </form>
      <div className="mission-intro mission-copy">
        <h1>Turn one goal into<br /><span>coordinated action.</span></h1>
        <p>Plan, research, remember, use tools, ask for approval, and deliver a reviewed result through one visible workflow.</p>
      </div>
      <div className="segmented mode-rail" aria-label="Mission mode">{['auto', 'research', 'write', 'review'].map(item => <button type="button" className={mode === item ? 'selected' : ''} onClick={() => setMode(item)} key={item}>{item === 'auto' ? <Sparkles size={13} /> : item === 'research' ? <Search size={13} /> : item === 'write' ? <FileText size={13} /> : <ShieldCheck size={13} />}{item}</button>)}</div>
      <div className="prompt-suggestions">{suggestions.map((item, index) => <button key={item} onClick={() => setGoal(item)}><span>0{index + 1}</span>{item}<ChevronRight size={14} /></button>)}</div>
      <div className="stage-launchers">
        {navigation.slice(1, 7).map(item => <button key={item.id} onClick={() => onNavigate(item.id)} className={`tone-${item.tone}`}><span className="nav-symbol"><item.icon size={19} /></span><strong>{item.label}</strong><small>{item.detail}</small></button>)}
      </div>
    </section>

    <section className="system-pulse">
      <div className="pulse-heading"><div><p className="eyebrow">Live architecture</p><h2>One coordinator, with specialists when the work needs them.</h2></div><span className="latency-signal"><i /> system synchronized</span></div>
      <div className="pulse-layout"><AgentTopology /><div className="metric-strip">
        <Metric label="Active" value={analytics.active_runs} detail={analytics.active_runs ? 'Processing now' : 'Queue clear'} tone="teal" />
        <Metric label="Success" value={`${analytics.total_runs ? Math.round(analytics.completed_runs / analytics.total_runs * 100) : 0}%`} detail={`${analytics.completed_runs} completed`} />
        <Metric label="Model calls" value={analytics.total_llm_calls} detail={`${compact(analytics.total_tokens)} tokens`} />
        <Metric label="P95 runtime" value={`${analytics.p95_execution_seconds.toFixed(1)}s`} detail="Last 1,000 runs" tone="amber" />
      </div></div>
    </section>

    <section className="data-section recent-runs"><div className="section-header"><div><p className="eyebrow">Recent work</p><h2>What your team has been doing</h2></div><span>{runs.length} total runs</span></div><RunTable runs={runs.slice(0, 6)} onSelect={onSelect} /></section>
  </div>
}

function RunsView({ session, runs, selected, onSelect, onRefresh }: { session: Session; runs: Run[]; selected: Run | null; onSelect: (id: string) => void; onRefresh: () => void }) {
  const [filter, setFilter] = useState('all')
  const shown = filter === 'all' ? runs : runs.filter(run => run.status === filter)
  return <><PageHeading eyebrow="A clear history" title="Activity" description="Follow each goal from first step to final result, including handoffs, time, and the choices made along the way." actions={<button className="secondary-button" onClick={onRefresh}><RefreshCw size={16} />Refresh</button>} />
    <div className="filter-row">{['all', 'queued', 'running', 'completed', 'failed', 'cancelled'].map(item => <button className={filter === item ? 'active' : ''} onClick={() => setFilter(item)} key={item}>{item}</button>)}</div>
    <div className="runs-layout"><section className="data-section"><RunTable runs={shown} onSelect={onSelect} selectedId={selected?.id} /></section><RunInspector run={selected} token={session.access_token} onChanged={onRefresh} /></div>
  </>
}

function RunInspector({ run, token, onChanged }: { run: Run | null; token: string; onChanged: () => void }) {
  if (!run) return <aside className="inspector empty-state"><Network size={30} /><h3>Select a run</h3><p>Its trace, timing, usage, and result will appear here.</p></aside>
  const active = ['queued', 'running'].includes(run.status)
  return <aside className="inspector"><div className="inspector-head"><div><Status status={run.status} /><h2>{run.goal}</h2><code>{run.id}</code></div>{active && <button className="danger-icon" title="Cancel run" onClick={async () => { await api.cancelRun(token, run.id); onChanged() }}><CircleStop size={19} /></button>}</div>
    <div className="progress-track"><span style={{ width: `${Math.max(2, run.progress * 100)}%` }} /></div>
    <div className="run-facts"><span><small>Queue</small>{run.queue_seconds.toFixed(2)}s</span><span><small>Execute</small>{run.execution_seconds.toFixed(2)}s</span><span><small>LLM calls</small>{run.llm_calls}</span><span><small>Tokens</small>{compact(run.input_tokens + run.output_tokens)}</span></div>
    <div className="trace"><h3>Execution trace</h3>{run.events?.map(event => <div className="trace-item" key={event.id}><span className={`trace-marker ${event.status ?? ''}`} /><div><strong>{event.message}</strong><small>{event.node ?? event.event_type} · {time(event.created_at)}</small></div></div>)}</div>
    {run.result_text && <div className="result-output"><h3>Final result</h3><p>{run.result_text}</p></div>}
    {run.error_message && <div className="form-error"><CircleAlert size={16} />{run.error_message}</div>}
  </aside>
}

function KnowledgeView({ session, documents, refresh }: { session: Session; documents: Document[]; refresh: (quiet?: boolean) => Promise<void> }) {
  const fileRef = useRef<HTMLInputElement>(null); const [busy, setBusy] = useState(false)
  async function upload(file?: File) { if (!file) return; setBusy(true); try { await api.uploadDocument(session.access_token, session.workspace.id, file); await refresh() } finally { setBusy(false); if (fileRef.current) fileRef.current.value = '' } }
  return <><PageHeading eyebrow="Sources you trust" title="Library" description="Add reference documents so answers can be grounded in your material. Every source stays isolated to this workspace." actions={<><input ref={fileRef} hidden type="file" accept="application/pdf" onChange={e => void upload(e.target.files?.[0])} /><button className="primary-button" onClick={() => fileRef.current?.click()} disabled={busy}>{busy ? <RefreshCw size={17} className="spinning" /> : <Upload size={17} />}Upload PDF</button></>} />
    <div className="knowledge-summary"><div><Database size={25} /><span><strong>{documents.filter(d => d.status === 'indexed').length}</strong> indexed sources</span></div><div><Archive size={25} /><span><strong>{documents.reduce((sum, d) => sum + d.chunk_count, 0)}</strong> retrievable chunks</span></div><div><ShieldCheck size={25} /><span><strong>Workspace</strong> isolation</span></div></div>
    <section className="data-section"><div className="section-header"><div><p className="eyebrow">Source registry</p><h2>Documents</h2></div></div>{documents.length ? <div className="document-list">{documents.map(doc => <div className="document-row" key={doc.id}><div className="file-icon"><FileText size={20} /></div><div><strong>{doc.filename}</strong><small>{bytes(doc.size_bytes)} · version {doc.version} · {doc.chunk_count} chunks</small></div><Status status={doc.status} /><button className="icon-button" title="Delete document" onClick={async () => { await api.deleteDocument(session.access_token, doc.id); await refresh() }}><Trash2 size={17} /></button></div>)}</div> : <Empty icon={FileText} title="No knowledge sources" text="Upload a PDF to create a workspace-scoped retrieval source." />}</section>
  </>
}

function MemoryView({ session, memories, refresh }: { session: Session; memories: MemoryRecord[]; refresh: (quiet?: boolean) => Promise<void> }) {
  const [content, setContent] = useState(''); const [type, setType] = useState('preference')
  async function add(event: FormEvent) { event.preventDefault(); await api.createMemory(session.access_token, session.workspace.id, type, content); setContent(''); await refresh() }
  return <><PageHeading eyebrow="Context that carries forward" title="Memory" description="Keep the preferences and facts that make future work more personal. You can always see, edit, or remove what is remembered." />
    <section className="surface memory-editor"><div><p className="eyebrow">Add record</p><h2>Teach the workspace one durable fact</h2></div><form onSubmit={add}><select value={type} onChange={e => setType(e.target.value)}><option value="preference">Preference</option><option value="profile">Profile</option><option value="project">Project</option><option value="constraint">Constraint</option></select><input value={content} onChange={e => setContent(e.target.value)} required minLength={2} maxLength={4000} placeholder="Example: Prefer concise recommendations with explicit tradeoffs." /><button className="primary-button"><Plus size={17} />Add</button></form></section>
    <div className="memory-grid">{memories.map(item => <article className="memory-item" key={item.id}><div><span className={`type-mark ${item.memory_type}`}><Brain size={15} />{item.memory_type}</span><button className="icon-button" title="Forget memory" onClick={async () => { await api.deleteMemory(session.access_token, item.id); await refresh() }}><Trash2 size={16} /></button></div><p>{item.content}</p><small>Updated {date(item.updated_at)}</small></article>)}{!memories.length && <Empty icon={Brain} title="Memory is empty" text="Add only facts that should influence more than one run." />}</div>
  </>
}

function ToolboxView({ session, tools, refresh }: { session: Session; tools: Tool[]; refresh: (quiet?: boolean) => Promise<void> }) {
  return <><PageHeading eyebrow="What your team can use" title="Capabilities" description="Choose which abilities are available. Sensitive actions still wait for your explicit approval." />
    <section className="data-section"><div className="tool-list">{tools.map(tool => <div className="tool-row" key={tool.name}><div className="tool-symbol"><Wrench size={19} /></div><div><strong>{tool.name}</strong><p>{tool.description}</p><span className={`permission ${tool.permission}`}>{tool.permission.replaceAll('_', ' ')}</span>{tool.requires_confirmation && <span className="permission approval">approval required</span>}</div><label className="switch"><input type="checkbox" checked={tool.enabled} onChange={async e => { await api.updateTool(session.access_token, session.workspace.id, tool.name, e.target.checked); await refresh(true) }} /><span /></label></div>)}</div></section>
  </>
}

function ApprovalsView({ session, approvals, refresh }: { session: Session; approvals: Approval[]; refresh: (quiet?: boolean) => Promise<void> }) {
  const pending = approvals.filter(item => item.status === 'pending'); const decided = approvals.filter(item => item.status !== 'pending')
  async function decide(id: string, decision: 'approved' | 'rejected') { await api.decideApproval(session.access_token, id, decision); await refresh() }
  return <><PageHeading eyebrow="You remain in control" title="Reviews" description="Review the exact action before anything sensitive happens. Each approval applies once, to the version you can see." />
    <div className="approval-banner"><ShieldCheck size={25} /><div><strong>{pending.length} actions waiting</strong><span>Nothing sensitive will happen until you decide.</span></div></div>
    <section className="data-section"><div className="section-header"><div><p className="eyebrow">Decision queue</p><h2>Pending</h2></div></div>{pending.length ? pending.map(item => <div className="approval-row" key={item.id}><div><span className="risk-label">Review required</span><h3>{item.summary}</h3><small>{item.action_type} · run {item.run_id.slice(-8)}</small><pre>{JSON.stringify(item.proposal, null, 2)}</pre></div><div><button className="secondary-button" onClick={() => void decide(item.id, 'rejected')}><XCircle size={16} />Reject</button><button className="primary-button" onClick={() => void decide(item.id, 'approved')}><Check size={16} />Approve once</button></div></div>) : <Empty icon={ShieldCheck} title="Approval queue clear" text="No agent action is currently waiting for human authority." />}</section>
    {decided.length > 0 && <section className="data-section"><div className="section-header"><div><p className="eyebrow">Audit history</p><h2>Decisions</h2></div></div>{decided.map(item => <div className="compact-row" key={item.id}><Status status={item.status} /><strong>{item.summary}</strong><small>{date(item.decided_at ?? item.created_at)}</small></div>)}</section>}
  </>
}

function InsightsView({ analytics, runs }: { analytics: Analytics; runs: Run[] }) {
  const max = Math.max(1, ...analytics.daily_runs.map(item => item.runs))
  return <><PageHeading eyebrow="Learn from the work" title="Patterns" description="A simple view of speed, reliability, and usage helps you understand where the system is serving you well." />
    <div className="metric-strip"><Metric label="Total runs" value={analytics.total_runs} detail={`${analytics.active_runs} active`} /><Metric label="Average runtime" value={`${analytics.average_execution_seconds.toFixed(1)}s`} detail={`P95 ${analytics.p95_execution_seconds.toFixed(1)}s`} tone="teal" /><Metric label="Token volume" value={compact(analytics.total_tokens)} detail={`${analytics.total_llm_calls} provider calls`} /><Metric label="Tool calls" value={analytics.total_tool_calls} detail={`${Math.round(analytics.approval_rate * 100)}% approval rate`} tone="amber" /></div>
    <div className="insight-grid"><section className="surface chart-surface"><div className="section-title"><div><p className="eyebrow">Throughput</p><h2>Runs over time</h2></div><Activity size={20} /></div><div className="bar-chart">{analytics.daily_runs.length ? analytics.daily_runs.map(item => <div className="bar-column" key={item.date}><span style={{ height: `${Math.max(8, item.runs / max * 100)}%` }} title={`${item.runs} runs`} /><small>{item.date.slice(5)}</small></div>) : <p className="chart-empty">Run history will form the chart.</p>}</div></section><section className="surface health-surface"><div className="section-title"><div><p className="eyebrow">Lifecycle mix</p><h2>Run health</h2></div><Gauge size={20} /></div>{Object.entries(analytics.status_counts).map(([status, count]) => <div className="health-row" key={status}><Status status={status} /><div><span style={{ width: `${count / Math.max(1, analytics.total_runs) * 100}%` }} /></div><strong>{count}</strong></div>)}</section></div>
    <section className="data-section"><div className="section-header"><div><p className="eyebrow">Slow-path review</p><h2>Longest runs</h2></div></div><RunTable runs={[...runs].sort((a, b) => b.execution_seconds - a.execution_seconds).slice(0, 5)} onSelect={() => undefined} /></section>
  </>
}

function SettingsView({ session, onLogout }: { session: Session; onLogout: () => void }) {
  return <><PageHeading eyebrow="Your space, your boundaries" title="Settings" description="Manage your identity, workspace details, and the local environment behind this experience." />
    <div className="settings-layout"><section className="surface settings-section"><div className="section-title"><div><p className="eyebrow">Operator</p><h2>Account identity</h2></div><Users size={20} /></div><dl><dt>Name</dt><dd>{session.user.display_name}</dd><dt>Email</dt><dd>{session.user.email}</dd><dt>Role</dt><dd>{session.user.role}</dd><dt>User ID</dt><dd><code>{session.user.id}</code></dd></dl></section><section className="surface settings-section"><div className="section-title"><div><p className="eyebrow">Workspace</p><h2>Runtime boundary</h2></div><Database size={20} /></div><dl><dt>Name</dt><dd>{session.workspace.name}</dd><dt>Workspace ID</dt><dd><code>{session.workspace.id}</code></dd><dt>API contract</dt><dd>/api/v1</dd><dt>Environment</dt><dd><span className="environment">Local development</span></dd></dl></section></div>
    <section className="danger-zone"><div><h2>End local session</h2><p>The browser token is removed. Durable workspace data remains in the platform database.</p></div><button className="secondary-button" onClick={onLogout}><LogOut size={17} />Sign out</button></section>
  </>
}

function RunTable({ runs, onSelect, selectedId }: { runs: Run[]; onSelect: (id: string) => void; selectedId?: string }) {
  if (!runs.length) return <Empty icon={Network} title="No runs yet" text="Launch a mission to create the first durable execution record." />
  return <div className="table-wrap"><table><thead><tr><th>Mission</th><th>Status</th><th>Mode</th><th>Calls</th><th>Runtime</th><th>Created</th><th /></tr></thead><tbody>{runs.map(run => <tr key={run.id} className={selectedId === run.id ? 'selected' : ''} onClick={() => onSelect(run.id)}><td><strong>{run.goal}</strong><small>{run.id.slice(-10)}</small></td><td><Status status={run.status} /></td><td className="capitalize">{run.mode}</td><td>{run.llm_calls}</td><td>{run.execution_seconds ? `${run.execution_seconds.toFixed(1)}s` : '—'}</td><td>{date(run.created_at)}</td><td><ChevronRight size={17} /></td></tr>)}</tbody></table></div>
}

function AgentTopology() { return <div className="agent-topology"><div className="topology-row"><span className="agent-node manager"><Bot size={20} /><b>Coordinator</b><small>understands and guides</small></span></div><div className="connector"><i /><i /><i /></div><div className="topology-row specialists"><span className="agent-node"><Search size={18} /><b>Researcher</b><small>finds evidence</small></span><span className="agent-node"><FileText size={18} /><b>Writer</b><small>shapes the result</small></span><span className="agent-node"><ShieldCheck size={18} /><b>Reviewer</b><small>checks the work</small></span></div></div> }
function PageHeading({ eyebrow, title, description, actions }: { eyebrow: string; title: string; description: string; actions?: React.ReactNode }) { return <div className="page-heading"><div><p className="eyebrow">{eyebrow}</p><h1>{title}</h1><p>{description}</p></div>{actions && <div className="page-actions">{actions}</div>}</div> }
function Metric({ label, value, detail, tone = 'default' }: { label: string; value: string | number; detail: string; tone?: string }) { return <div className={`metric ${tone}`}><span>{label}</span><strong>{value}</strong><small>{detail}</small></div> }
function Status({ status }: { status: string }) { const clean = String(status).replace('RunStatus.', '').toLowerCase(); return <span className={`status status-${clean}`}><i />{clean.replaceAll('_', ' ')}</span> }
function Empty({ icon: Icon, title, text }: { icon: typeof Activity; title: string; text: string }) { return <div className="empty-state"><Icon size={30} /><h3>{title}</h3><p>{text}</p></div> }
function message(error: unknown) { return error instanceof Error ? error.message : 'Something went wrong.' }
function initials(name: string) { return name.split(/\s+/).map(part => part[0]).join('').slice(0, 2).toUpperCase() }
function firstName(name: string) { return name.trim().split(/\s+/)[0] || 'there' }
function daypart() { const hour = new Date().getHours(); return hour < 12 ? 'morning' : hour < 17 ? 'afternoon' : 'evening' }
function compact(value: number) { return new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(value) }
function date(value: string) { return new Intl.DateTimeFormat('en', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }).format(new Date(value)) }
function time(value: string) { return new Intl.DateTimeFormat('en', { hour: 'numeric', minute: '2-digit', second: '2-digit' }).format(new Date(value)) }
function bytes(value: number) { return value > 1024 * 1024 ? `${(value / 1024 / 1024).toFixed(1)} MB` : `${Math.ceil(value / 1024)} KB` }
