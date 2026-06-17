import { useState, useEffect, useRef } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

// Production-style tenant context (comes from "login" or real auth later)
function getStoredTenant() {
  const stored = localStorage.getItem('mmfsa_tenant')
  if (stored) return JSON.parse(stored)
  // Default demo tenant for first run
  return {
    tenant_id: 'constr-42',
    industry: 'construction',
    company_name: 'Demo Construction Co'
  }
}

function App() {
  const [tenant, setTenant] = useState(getStoredTenant())
  const [isLoggedIn, setIsLoggedIn] = useState(!!localStorage.getItem('mmfsa_tenant'))
  const [showLogin, setShowLogin] = useState(!localStorage.getItem('mmfsa_tenant'))

  // Login form state
  const [loginTenantId, setLoginTenantId] = useState('constr-42')
  const [loginIndustry, setLoginIndustry] = useState('construction')
  const [loginCompany, setLoginCompany] = useState('Demo Construction Co')

  // Main app state
  const [sessionId] = useState('react-' + Date.now())
  const [messages, setMessages] = useState([])
  const [guidance, setGuidance] = useState(null)
  const [loading, setLoading] = useState(false)
  const [capturedImage, setCapturedImage] = useState(null) // base64 without prefix
  const [isStreaming, setIsStreaming] = useState(false)
  const [uploadStatus, setUploadStatus] = useState('')

  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const chatRef = useRef(null)

  // Persist tenant
  useEffect(() => {
    if (isLoggedIn) {
      localStorage.setItem('mmfsa_tenant', JSON.stringify(tenant))
    }
  }, [tenant, isLoggedIn])

  // Auto scroll chat
  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight
    }
  }, [messages])

  // === Auth / Tenant handling (prototype for now) ===
  const handleLogin = () => {
    const newTenant = {
      tenant_id: loginTenantId.trim() || 'demo-tenant-001',
      industry: loginIndustry,
      company_name: loginCompany.trim() || 'Unknown Company'
    }
    setTenant(newTenant)
    setIsLoggedIn(true)
    setShowLogin(false)
    localStorage.setItem('mmfsa_tenant', JSON.stringify(newTenant))
    addMessage('System', `Logged in as ${newTenant.company_name} (${newTenant.industry})`, 'system')
  }

  const handleLogout = () => {
    localStorage.removeItem('mmfsa_tenant')
    setIsLoggedIn(false)
    setTenant(null)
    setMessages([])
    setGuidance(null)
    setCapturedImage(null)
    setShowLogin(true)
  }

  const switchTenant = () => {
    setShowLogin(true)
  }

  // === API helper (always sends production headers) ===
  const apiCall = async (path, options = {}) => {
    const headers = {
      'Content-Type': 'application/json',
      'X-Tenant-ID': tenant.tenant_id,
      'X-Industry': tenant.industry,
      ...(options.headers || {})
    }

    const res = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers
    })

    if (!res.ok) {
      const text = await res.text().catch(() => '')
      throw new Error(`HTTP ${res.status}: ${text || res.statusText}`)
    }
    return res.json()
  }

  // === Messages ===
  const addMessage = (role, content, type = 'user') => {
    const msg = {
      id: Date.now() + Math.random(),
      role,
      content,
      type,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
    setMessages(prev => [...prev, msg])
    return msg
  }

  // === Camera ===
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' }
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        setIsStreaming(true)
      }
    } catch (err) {
      alert('Camera error: ' + err.message)
    }
  }

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    setIsStreaming(false)
  }

  const capturePhoto = () => {
    const video = videoRef.current
    if (!video || !video.videoWidth) return alert('Start camera first')

    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0)

    const dataUrl = canvas.toDataURL('image/jpeg', 0.85)
    const base64 = dataUrl.split(',')[1]
    setCapturedImage(base64)

    stopCamera()
    addMessage('You', 'Photo captured', 'system')
  }

  const clearPhoto = () => {
    setCapturedImage(null)
  }

  // === Voice (Web Speech API) ===
  const startVoice = () => {
    if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
      return alert('Voice recognition not supported in this browser')
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    const recognition = new SpeechRecognition()
    recognition.lang = 'en-US'
    recognition.interimResults = false
    recognition.maxAlternatives = 1

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript
      addMessage('You', transcript)
      processQuery(transcript)
    }

    recognition.onerror = (e) => {
      alert('Voice error: ' + e.error)
    }

    recognition.start()
  }

  // === Core query flow (production headers) ===
  const processQuery = async (text) => {
    if (!text.trim() || !tenant) return

    setLoading(true)
    setGuidance(null)

    try {
      const payload = {
        session_id: sessionId,
        transcript: text,
        image_b64: capturedImage || null
      }

      const data = await apiCall('/get-guidance', {
        method: 'POST',
        body: JSON.stringify(payload)
      })

      // Show in chat
      addMessage('Assistant', data.immediate_response || 'Guidance received')

      // Rich guidance panel data
      setGuidance({
        ...data,
        tenant_id: tenant.tenant_id,
        industry: tenant.industry
      })

      // Clear photo after use (optional)
      // setCapturedImage(null)
    } catch (err) {
      console.error(err)
      addMessage('System', 'Error: ' + err.message, 'error')
      setGuidance({
        immediate_response: 'Failed to reach backend',
        disclaimer: 'Could not connect to ' + API_BASE + '. Check your .env and ngrok.'
      })
    } finally {
      setLoading(false)
    }
  }

  const sendText = () => {
    const input = document.getElementById('text-input')
    if (!input || !input.value.trim()) return
    const text = input.value.trim()
    addMessage('You', text)
    processQuery(text)
    input.value = ''
  }

  // === Document Upload (re-added, production headers) ===
  const handleUpload = async (e) => {
    const file = e.target.files[0]
    if (!file || !tenant) return

    setUploadStatus('Uploading...')

    const formData = new FormData()
    formData.append('file', file)
    formData.append('description', 'Uploaded via React frontend')

    try {
      const res = await fetch(`${API_BASE}/upload/document`, {
        method: 'POST',
        headers: {
          'X-Tenant-ID': tenant.tenant_id,
          'X-Industry': tenant.industry
          // Note: do NOT set Content-Type for FormData
        },
        body: formData
      })

      const data = await res.json()

      if (data.success) {
        setUploadStatus(`✅ Uploaded: ${data.filename} for ${data.tenant_id}`)
        addMessage('System', `Manual uploaded: ${data.filename} (now available in RAG for this tenant)`, 'system')
      } else {
        setUploadStatus('Upload failed')
      }
    } catch (err) {
      setUploadStatus('Error: ' + err.message)
    }

    // Reset file input
    e.target.value = ''
    setTimeout(() => setUploadStatus(''), 4000)
  }

  // === Demo helpers ===
  const runDemo = (text) => {
    addMessage('You', text)
    processQuery(text)
  }

  if (showLogin || !isLoggedIn) {
    return (
      <div className="min-h-screen bg-slate-950 text-white flex items-center justify-center p-6">
        <div className="max-w-md w-full bg-slate-900 border border-slate-700 rounded-3xl p-8">
          <div className="text-center mb-8">
            <div className="w-16 h-16 bg-emerald-500 rounded-2xl mx-auto flex items-center justify-center mb-4">
              <i className="fa-solid fa-wrench text-3xl"></i>
            </div>
            <h1 className="text-3xl font-semibold tracking-tighter">MM-FSA</h1>
            <p className="text-slate-400 mt-1">Field Service Assistant</p>
            <p className="text-xs text-amber-400 mt-2">Production SaaS Demo • Tenant via Headers</p>
          </div>

          <div className="space-y-4">
            <div>
              <label className="text-sm text-slate-400 block mb-1">Tenant ID</label>
              <input
                value={loginTenantId}
                onChange={(e) => setLoginTenantId(e.target.value)}
                className="w-full bg-slate-800 border border-slate-600 rounded-xl px-4 py-3 font-mono text-sm"
                placeholder="constr-42"
              />
            </div>

            <div>
              <label className="text-sm text-slate-400 block mb-1">Industry</label>
              <select
                value={loginIndustry}
                onChange={(e) => setLoginIndustry(e.target.value)}
                className="w-full bg-slate-800 border border-slate-600 rounded-xl px-4 py-3 text-sm"
              >
                <option value="construction">Construction</option>
                <option value="water_treatment">Water Treatment</option>
                <option value="oil_gas">Oil &amp; Gas</option>
                <option value="manufacturing">Manufacturing</option>
                <option value="general_industrial">General Industrial</option>
              </select>
            </div>

            <div>
              <label className="text-sm text-slate-400 block mb-1">Company Name</label>
              <input
                value={loginCompany}
                onChange={(e) => setLoginCompany(e.target.value)}
                className="w-full bg-slate-800 border border-slate-600 rounded-xl px-4 py-3"
                placeholder="Acme Construction"
              />
            </div>
          </div>

          <button
            onClick={handleLogin}
            className="mt-6 w-full bg-emerald-600 hover:bg-emerald-500 py-3.5 rounded-2xl font-semibold text-lg"
          >
            Sign In (Prototype)
          </button>

          <p className="text-[10px] text-center text-slate-500 mt-4">
            In production this would be a real login with JWT that contains tenant claims.
            Right now it just sets the headers the backend expects.
          </p>
        </div>
      </div>
    )
  }

  // === Main App ===
  return (
    <div className="min-h-screen bg-slate-950 text-slate-200">
      {/* Top Nav - Production feel */}
      <nav className="border-b border-slate-800 bg-slate-900">
        <div className="max-w-screen-2xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-x-3">
            <div className="w-9 h-9 bg-emerald-500 rounded-2xl flex items-center justify-center">
              <i className="fa-solid fa-wrench text-white"></i>
            </div>
            <div>
              <span className="font-semibold text-2xl tracking-tighter">MM-FSA</span>
              <span className="ml-2 text-xs px-2 py-0.5 bg-slate-800 rounded text-emerald-400">SaaS</span>
            </div>
          </div>

          <div className="flex items-center gap-x-4 text-sm">
            {/* Current Tenant */}
            <div className="flex items-center gap-x-2 bg-slate-800 border border-slate-700 rounded-2xl px-4 py-1.5">
              <i className="fa-solid fa-building text-amber-400"></i>
              <div>
                <div className="font-mono text-amber-300 text-xs">{tenant.tenant_id}</div>
                <div className="text-[10px] text-slate-400">{tenant.company_name} • {tenant.industry}</div>
              </div>
            </div>

            <button
              onClick={switchTenant}
              className="px-4 py-2 text-xs bg-slate-800 hover:bg-slate-700 rounded-2xl border border-slate-600"
            >
              Switch Tenant
            </button>

            <button
              onClick={handleLogout}
              className="px-4 py-2 text-xs text-red-400 hover:text-red-300"
            >
              Logout
            </button>
          </div>
        </div>
      </nav>

      <div className="max-w-screen-2xl mx-auto px-6 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: Camera + Voice + Upload */}
          <div className="lg:col-span-5 space-y-6">
            {/* Camera */}
            <div className="bg-slate-900 border border-slate-700 rounded-3xl p-5">
              <div className="flex items-center justify-between mb-4">
                <div className="font-semibold flex items-center gap-x-2">
                  <i className="fa-solid fa-camera text-emerald-400"></i>
                  <span>Camera</span>
                </div>
                {capturedImage && (
                  <button onClick={clearPhoto} className="text-xs text-red-400">Clear photo</button>
                )}
              </div>

              <div className="relative bg-black rounded-2xl overflow-hidden aspect-video border border-slate-700">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  className={`w-full h-full object-cover ${isStreaming ? '' : 'hidden'}`}
                />
                {!isStreaming && !capturedImage && (
                  <div className="absolute inset-0 flex items-center justify-center text-slate-500">
                    <div className="text-center">
                      <i className="fa-solid fa-camera-retro text-5xl mb-3 block"></i>
                      <p className="text-sm">Point camera at equipment</p>
                    </div>
                  </div>
                )}
                {capturedImage && (
                  <img
                    src={`data:image/jpeg;base64,${capturedImage}`}
                    className="w-full h-full object-cover"
                    alt="Captured"
                  />
                )}
              </div>

              <div className="flex gap-3 mt-4">
                {!isStreaming ? (
                  <button onClick={startCamera} className="flex-1 py-3 bg-emerald-600 hover:bg-emerald-500 rounded-2xl font-medium">
                    Start Camera
                  </button>
                ) : (
                  <button onClick={stopCamera} className="flex-1 py-3 bg-slate-700 hover:bg-slate-600 rounded-2xl font-medium">
                    Stop Camera
                  </button>
                )}
                <button
                  onClick={capturePhoto}
                  disabled={!isStreaming}
                  className="flex-1 py-3 bg-slate-800 border border-slate-600 rounded-2xl font-medium disabled:opacity-50"
                >
                  Capture
                </button>
              </div>
            </div>

            {/* Voice + Upload */}
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-slate-900 border border-slate-700 rounded-3xl p-5">
                <div className="font-semibold mb-3 flex items-center gap-x-2">
                  <i className="fa-solid fa-microphone text-purple-400"></i>
                  <span>Voice</span>
                </div>
                <button
                  onClick={startVoice}
                  className="w-full py-3 bg-purple-600 hover:bg-purple-500 rounded-2xl font-medium"
                >
                  Speak
                </button>
                <p className="text-[10px] text-slate-500 mt-2">Web Speech API</p>
              </div>

              {/* Upload Manual - Re-added */}
              <div className="bg-slate-900 border border-slate-700 rounded-3xl p-5">
                <div className="font-semibold mb-3 flex items-center gap-x-2">
                  <i className="fa-solid fa-file-upload text-blue-400"></i>
                  <span>Upload Manual</span>
                </div>
                <label className="block w-full py-3 bg-slate-800 border border-slate-600 hover:bg-slate-700 rounded-2xl text-center cursor-pointer font-medium">
                  Choose PDF / MD / TXT
                  <input
                    type="file"
                    accept=".pdf,.md,.txt,.markdown"
                    onChange={handleUpload}
                    className="hidden"
                  />
                </label>
                {uploadStatus && (
                  <p className="text-xs mt-2 text-emerald-400">{uploadStatus}</p>
                )}
                <p className="text-[10px] text-slate-500 mt-2">Goes to this tenant only</p>
              </div>
            </div>
          </div>

          {/* Right Column: Chat + Guidance */}
          <div className="lg:col-span-7 space-y-6">
            {/* Chat */}
            <div className="bg-slate-900 border border-slate-700 rounded-3xl flex flex-col h-[380px]">
              <div className="px-5 py-4 border-b border-slate-700 flex items-center justify-between">
                <span className="font-semibold">Conversation</span>
                <span className="text-xs font-mono text-emerald-400">{sessionId}</span>
              </div>
              <div ref={chatRef} className="flex-1 overflow-auto p-5 text-sm space-y-3">
                {messages.length === 0 && (
                  <div className="text-slate-500 text-sm">Start by speaking, typing, or using the demo buttons below.</div>
                )}
                {messages.map(msg => (
                  <div key={msg.id} className={msg.type === 'system' ? 'text-xs text-emerald-400' : ''}>
                    <span className="font-medium mr-2">{msg.role}:</span>
                    {msg.content}
                  </div>
                ))}
              </div>
              <div className="p-4 border-t border-slate-700">
                <div className="flex gap-2">
                  <input
                    id="text-input"
                    type="text"
                    placeholder="Describe the issue or ask a question..."
                    className="flex-1 bg-slate-800 border border-slate-700 rounded-2xl px-4 py-3 text-sm focus:outline-none focus:border-emerald-500"
                    onKeyDown={(e) => e.key === 'Enter' && sendText()}
                  />
                  <button
                    onClick={sendText}
                    className="px-6 bg-emerald-600 hover:bg-emerald-500 rounded-2xl font-semibold"
                  >
                    Send
                  </button>
                </div>
              </div>
            </div>

            {/* Guidance Panel */}
            <div className="bg-slate-900 border border-slate-700 rounded-3xl p-5">
              <div className="flex items-center justify-between mb-4">
                <span className="font-semibold flex items-center gap-x-2">
                  <i className="fa-solid fa-robot text-amber-400"></i>
                  Agent Guidance
                </span>
                {guidance?.confidence && (
                  <span className="text-xs px-3 py-1 bg-slate-800 rounded-2xl">
                    {Math.round(guidance.confidence * 100)}%
                  </span>
                )}
              </div>

              {!guidance && !loading && (
                <div className="text-slate-500 text-sm py-8 text-center">
                  Responses will appear here. Try the demo buttons.
                </div>
              )}

              {loading && (
                <div className="flex items-center gap-3 py-6 text-emerald-400">
                  <div className="animate-spin h-5 w-5 border-2 border-emerald-400 border-t-transparent rounded-full"></div>
                  Talking to backend...
                </div>
              )}

              {guidance && (
                <div className="space-y-4 text-sm">
                  {/* Director / SaaS info */}
                  <div className="flex flex-wrap gap-2 text-xs">
                    <span className="px-3 py-1 bg-slate-800 rounded-2xl">
                      Industry: {guidance.query_industry || tenant.industry}
                    </span>
                    {guidance.was_cross_industry && (
                      <span className="px-3 py-1 bg-red-900/60 text-red-300 rounded-2xl">CROSS-INDUSTRY</span>
                    )}
                    {guidance.query_intent && (
                      <span className="px-3 py-1 bg-slate-700 rounded-2xl">{guidance.query_intent}</span>
                    )}
                  </div>

                  <div>
                    <div className="text-xs text-emerald-400 mb-1">RESPONSE</div>
                    <div className="font-medium">{guidance.immediate_response}</div>
                  </div>

                  {guidance.plan?.steps?.length > 0 && (
                    <div>
                      <div className="text-xs text-emerald-400 mb-1.5">STEPS</div>
                      <div className="space-y-2">
                        {guidance.plan.steps.map((step, i) => (
                          <div key={i} className="bg-slate-800 rounded-2xl p-3 text-sm">
                            {step.step_number}. {step.description}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {guidance.disclaimer && (
                    <div className="text-xs p-3 bg-amber-900/30 border border-amber-700 rounded-2xl">
                      {guidance.disclaimer}
                    </div>
                  )}

                  {guidance.plan?.citations?.length > 0 && (
                    <div className="text-[10px] text-slate-400">
                      Sources: {guidance.plan.citations.join(', ')}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Quick demos */}
            <div className="flex flex-wrap gap-2">
              <button onClick={() => runDemo('pump is vibrating badly and leaking from the seal')} className="text-xs px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-2xl border border-slate-600">
                Demo: Pump vibration (construction)
              </button>
              <button onClick={() => runDemo('what is hazard and operability study')} className="text-xs px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-2xl border border-slate-600">
                Demo: HAZOP (general)
              </button>
              <button onClick={() => runDemo('clarifier sludge handling procedure')} className="text-xs px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-2xl border border-slate-600">
                Demo: Clarifier (cross-industry test)
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
