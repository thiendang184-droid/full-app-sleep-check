import { useState } from "react";

const MOON_SVG = (
  <svg viewBox="0 0 100 100" width="120" height="120" style={{position:"absolute",top:-30,right:-20,opacity:0.08}}>
    <circle cx="50" cy="50" r="45" fill="#a78bfa"/>
    <circle cx="30" cy="30" r="18" fill="#0f0a1e"/>
  </svg>
);

const StarField = () => (
  <div style={{position:"fixed",inset:0,pointerEvents:"none",zIndex:0}}>
    {Array.from({length:60}).map((_,i)=>{
      const x = Math.random()*100, y = Math.random()*100;
      const size = Math.random()*2+0.5;
      const delay = Math.random()*4;
      return (
        <div key={i} style={{
          position:"absolute", left:`${x}%`, top:`${y}%`,
          width:size, height:size, borderRadius:"50%",
          background:"white",
          animation:`twinkle ${2+Math.random()*3}s ${delay}s infinite alternate`,
          opacity: Math.random()*0.7+0.1
        }}/>
      );
    })}
  </div>
);

const FIELDS = {
  gender: { label: "Giới tính", type: "select", options: ["Female","Male"], icon: "👤" },
  age: { label: "Tuổi", type: "number", min:18, max:80, icon: "🎂", unit:"tuổi" },
  sleep_duration: { label: "Số giờ ngủ / đêm", type: "number", min:4, max:12, step:0.5, icon: "🌙", unit:"giờ" },
  physical_activity: { label: "Hoạt động thể chất", type: "number", min:0, max:100, icon: "🏃", unit:"/100" },
  stress_level: { label: "Mức độ stress", type: "number", min:1, max:10, icon: "😓", unit:"/10" },
  bmi_category: { label: "Chỉ số BMI", type: "select", options: ["Normal","Normal Weight","Obese","Overweight"], icon: "⚖️" },
  heart_rate: { label: "Nhịp tim lúc nghỉ", type: "number", min:40, max:120, icon: "❤️", unit:"bpm" },
  daily_steps: { label: "Số bước chân / ngày", type: "number", min:0, max:30000, icon: "👟", unit:"bước" },
};

const DEFAULT_VALS = {
  gender:"Female", age:30, sleep_duration:7, physical_activity:50,
  stress_level:5, bmi_category:"Normal", heart_rate:70, daily_steps:7000
};

function ScoreRing({value, max, color, label, emoji}) {
  const pct = value/max;
  const r=36, circ=2*Math.PI*r;
  const dash = pct*circ;
  return (
    <div style={{display:"flex",flexDirection:"column",alignItems:"center",gap:8}}>
      <div style={{position:"relative",width:90,height:90}}>
        <svg width="90" height="90" style={{transform:"rotate(-90deg)"}}>
          <circle cx="45" cy="45" r={r} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="7"/>
          <circle cx="45" cy="45" r={r} fill="none" stroke={color} strokeWidth="7"
            strokeDasharray={`${dash} ${circ}`} strokeLinecap="round"
            style={{filter:`drop-shadow(0 0 6px ${color})`}}/>
        </svg>
        <div style={{position:"absolute",inset:0,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center"}}>
          <span style={{fontSize:22,fontWeight:800,color:"white",lineHeight:1}}>{value}</span>
          <span style={{fontSize:10,color:"rgba(255,255,255,0.5)"}}>/{max}</span>
        </div>
      </div>
      <div style={{textAlign:"center"}}>
        <div style={{fontSize:18}}>{emoji}</div>
        <div style={{fontSize:11,color:"rgba(255,255,255,0.6)",fontWeight:500,letterSpacing:0.5}}>{label}</div>
      </div>
    </div>
  );
}

function DisorderBadge({disorder}) {
  const map = {
    "None": {color:"#34d399",bg:"rgba(52,211,153,0.12)",label:"Không rối loạn",emoji:"✅"},
    "Insomnia": {color:"#f59e0b",bg:"rgba(245,158,11,0.12)",label:"Mất ngủ",emoji:"⚠️"},
    "Sleep Apnea": {color:"#f87171",bg:"rgba(248,113,113,0.12)",label:"Ngưng thở khi ngủ",emoji:"🚨"},
  };
  const s = map[disorder] || map["None"];
  return (
    <div style={{display:"flex",flexDirection:"column",alignItems:"center",gap:8,padding:"16px 24px",
      background:s.bg, border:`1px solid ${s.color}33`, borderRadius:16}}>
      <div style={{fontSize:32}}>{s.emoji}</div>
      <div style={{fontSize:13,fontWeight:700,color:s.color,letterSpacing:1,textTransform:"uppercase"}}>{s.label}</div>
      <div style={{fontSize:11,color:"rgba(255,255,255,0.45)"}}>Rối loạn giấc ngủ</div>
    </div>
  );
}

function buildPrompt(vals) {
  return `Bạn là chuyên gia phân tích giấc ngủ AI. Hãy dự đoán chất lượng giấc ngủ dựa trên thông tin sau và trả lời CHÍNH XÁC theo định dạng JSON (không thêm text nào khác):

Thông tin người dùng:
- Giới tính: ${vals.gender}
- Tuổi: ${vals.age}
- Số giờ ngủ/đêm: ${vals.sleep_duration}h
- Mức độ hoạt động thể chất: ${vals.physical_activity}/100
- Mức độ stress: ${vals.stress_level}/10
- BMI Category: ${vals.bmi_category}
- Nhịp tim lúc nghỉ: ${vals.heart_rate} bpm
- Số bước chân/ngày: ${vals.daily_steps}

Trả lời JSON với cấu trúc sau (không có markdown, chỉ JSON thuần):
{
  "quality": <số nguyên từ 4-9 đánh giá chất lượng giấc ngủ>,
  "disorder": "<một trong: None / Insomnia / Sleep Apnea>",
  "fatigue": <số nguyên từ 1-10 đánh giá mức độ mệt mỏi>,
  "advice": "<2-3 câu tư vấn cải thiện giấc ngủ bằng tiếng Việt, thực tế và cụ thể>"
}`;
}

export default function App() {
  const [vals, setVals] = useState(DEFAULT_VALS);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [step, setStep] = useState(0); // 0=form, 1=result

  const set = (k,v) => setVals(p=>({...p,[k]:v}));

  async function predict() {
    setLoading(true); setError(""); setResult(null);
    try {
      const res = await fetch("https://api.anthropic.com/v1/messages", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body: JSON.stringify({
          model:"claude-sonnet-4-20250514",
          max_tokens:1000,
          messages:[{role:"user", content: buildPrompt(vals)}]
        })
      });
      const data = await res.json();
      const text = data.content?.map(b=>b.text||"").join("").trim();
      const clean = text.replace(/```json|```/g,"").trim();
      const parsed = JSON.parse(clean);
      setResult(parsed);
      setStep(1);
    } catch(e) {
      setError("Lỗi phân tích. Vui lòng thử lại.");
    }
    setLoading(false);
  }

  const qualColor = result ? (result.quality>=7?"#34d399":result.quality>=5?"#f59e0b":"#f87171") : "#a78bfa";
  const fatColor = result ? (result.fatigue<=4?"#34d399":result.fatigue<=7?"#f59e0b":"#f87171") : "#a78bfa";

  return (
    <div style={{
      minHeight:"100vh", background:"#0a0614",
      fontFamily:"'DM Sans', 'Segoe UI', sans-serif",
      color:"white", position:"relative", overflowX:"hidden"
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800&family=Syne:wght@700;800&display=swap');
        @keyframes twinkle { from{opacity:0.1} to{opacity:0.8} }
        @keyframes fadeUp { from{opacity:0;transform:translateY(24px)} to{opacity:1;transform:translateY(0)} }
        @keyframes pulse { 0%,100%{transform:scale(1)} 50%{transform:scale(1.04)} }
        @keyframes spin { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
        @keyframes shimmer { 0%{background-position:-200% 0} 100%{background-position:200% 0} }
        ::-webkit-scrollbar{width:6px} ::-webkit-scrollbar-track{background:#0a0614}
        ::-webkit-scrollbar-thumb{background:#2d1b69;border-radius:3px}
        input[type=range]{-webkit-appearance:none;appearance:none;background:rgba(255,255,255,0.1);height:4px;border-radius:2px;outline:none;cursor:pointer}
        input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:#a78bfa;box-shadow:0 0 8px #a78bfa88;cursor:pointer}
        select{background:#13082a;border:1px solid rgba(167,139,250,0.25);color:white;padding:10px 14px;border-radius:10px;outline:none;width:100%;font-family:inherit;font-size:14px;cursor:pointer}
        select:focus{border-color:#a78bfa}
        option{background:#13082a}
        .field-card:hover{border-color:rgba(167,139,250,0.4)!important;transform:translateY(-1px)}
        .predict-btn:hover{transform:translateY(-2px);box-shadow:0 12px 40px rgba(167,139,250,0.4)!important}
        .predict-btn:active{transform:translateY(0)}
        .predict-btn:disabled{opacity:0.6;cursor:not-allowed}
      `}</style>

      <StarField/>

      {/* Glowing orbs */}
      <div style={{position:"fixed",top:-100,left:-100,width:500,height:500,borderRadius:"50%",
        background:"radial-gradient(circle,rgba(88,28,220,0.15) 0%,transparent 70%)",pointerEvents:"none",zIndex:0}}/>
      <div style={{position:"fixed",bottom:-100,right:-100,width:400,height:400,borderRadius:"50%",
        background:"radial-gradient(circle,rgba(167,139,250,0.1) 0%,transparent 70%)",pointerEvents:"none",zIndex:0}}/>

      <div style={{position:"relative",zIndex:1,maxWidth:680,margin:"0 auto",padding:"40px 20px 60px"}}>

        {/* Header */}
        <div style={{textAlign:"center",marginBottom:44,animation:"fadeUp 0.6s ease"}}>
          <div style={{fontSize:52,marginBottom:8}}>🌙</div>
          <h1 style={{
            fontFamily:"'Montserrat',sans-serif", fontSize:"clamp(26px,5vw,36px)",
            fontWeight:800, margin:0, letterSpacing:-0.5,
            background:"linear-gradient(135deg,#c4b5fd 0%,#a78bfa 40%,#818cf8 100%)",
            WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent"
          }}>Dự đoán chất lượng giấc ngủ</h1>
          <p style={{color:"rgba(255,255,255,0.45)",marginTop:10,fontSize:14,fontWeight:300,letterSpacing:0.3}}>
            Phân tích dựa trên Perceptron Network & AI
          </p>
        </div>

        {step === 0 ? (
          <div style={{animation:"fadeUp 0.7s ease 0.1s both"}}>
            {/* Form fields */}
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
              {Object.entries(FIELDS).map(([key,field])=>{
                const v = vals[key];
                return (
                  <div key={key} className="field-card" style={{
                    background:"rgba(255,255,255,0.03)", border:"1px solid rgba(255,255,255,0.08)",
                    borderRadius:16, padding:"16px 18px",
                    transition:"all 0.2s ease", gridColumn: key==="daily_steps"?"span 2":"span 1"
                  }}>
                    <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:12}}>
                      <span style={{fontSize:18}}>{field.icon}</span>
                      <span style={{fontSize:12,fontWeight:600,color:"rgba(255,255,255,0.5)",
                        letterSpacing:0.8,textTransform:"uppercase"}}>{field.label}</span>
                      {field.unit && <span style={{marginLeft:"auto",fontSize:12,color:"#a78bfa",fontWeight:700}}>{v}{field.unit}</span>}
                    </div>

                    {field.type==="select" ? (
                      <select value={v} onChange={e=>set(key,e.target.value)}>
                        {field.options.map(o=><option key={o}>{o}</option>)}
                      </select>
                    ) : (
                      <div>
                        <input type="range" min={field.min} max={field.max} step={field.step||1}
                          value={v} onChange={e=>set(key, field.step ? parseFloat(e.target.value) : parseInt(e.target.value))}
                          style={{width:"100%"}}/>
                        <div style={{display:"flex",justifyContent:"space-between",marginTop:4}}>
                          <span style={{fontSize:11,color:"rgba(255,255,255,0.25)"}}>{field.min}</span>
                          <span style={{fontSize:11,color:"rgba(255,255,255,0.25)"}}>{field.max}</span>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {error && <div style={{marginTop:16,padding:"12px 16px",background:"rgba(248,113,113,0.1)",
              border:"1px solid rgba(248,113,113,0.3)",borderRadius:10,color:"#f87171",fontSize:13}}>{error}</div>}

            <button className="predict-btn" onClick={predict} disabled={loading} style={{
              width:"100%", marginTop:24, padding:"18px",
              background:"linear-gradient(135deg,#7c3aed,#a78bfa)",
              border:"none", borderRadius:14, color:"white",
              fontSize:16, fontWeight:700, cursor:"pointer", letterSpacing:0.5,
              boxShadow:"0 8px 30px rgba(124,58,237,0.35)",
              transition:"all 0.2s ease", fontFamily:"inherit",
              display:"flex",alignItems:"center",justifyContent:"center",gap:12
            }}>
              {loading ? (
                <>
                  <div style={{width:20,height:20,border:"2px solid rgba(255,255,255,0.3)",
                    borderTopColor:"white",borderRadius:"50%",animation:"spin 0.8s linear infinite"}}/>
                  Đang phân tích...
                </>
              ) : (
                <><span>🔮</span> Phân tích giấc ngủ của tôi</>
              )}
            </button>
          </div>
        ) : result ? (
          <div style={{animation:"fadeUp 0.6s ease"}}>
            {/* Result card */}
            <div style={{
              background:"linear-gradient(135deg,rgba(124,58,237,0.08) 0%,rgba(167,139,250,0.05) 100%)",
              border:"1px solid rgba(167,139,250,0.2)", borderRadius:24, padding:"32px 28px",
              marginBottom:20, position:"relative",overflow:"hidden"
            }}>
              <div style={{position:"absolute",top:-40,right:-40,width:200,height:200,borderRadius:"50%",
                background:"radial-gradient(circle,rgba(167,139,250,0.08),transparent)",pointerEvents:"none"}}/>

              <h2 style={{fontFamily:"'Syne',sans-serif",fontSize:18,fontWeight:800,
                color:"rgba(255,255,255,0.9)",marginTop:0,marginBottom:28,letterSpacing:-0.3,textAlign:"center"}}>
                📊 Kết quả phân tích
              </h2>

              <div style={{display:"flex",justifyContent:"space-around",alignItems:"flex-start",flexWrap:"wrap",gap:20,marginBottom:28}}>
                <ScoreRing value={result.quality} max={10} color={qualColor} label="CHẤT LƯỢNG NGỦ" emoji="⭐"/>
                <DisorderBadge disorder={result.disorder}/>
                <ScoreRing value={result.fatigue} max={10} color={fatColor} label="MỆT MỎI" emoji="⚡"/>
              </div>

              {/* Overall rating bar */}
              <div style={{background:"rgba(255,255,255,0.04)",borderRadius:12,padding:"14px 18px"}}>
                <div style={{display:"flex",justifyContent:"space-between",marginBottom:8}}>
                  <span style={{fontSize:12,color:"rgba(255,255,255,0.5)",fontWeight:600,letterSpacing:0.5}}>TỔNG QUAN</span>
                  <span style={{fontSize:12,fontWeight:700,color:qualColor}}>
                    {result.quality>=7?"Tốt":result.quality>=5?"Trung bình":"Kém"}
                  </span>
                </div>
                <div style={{height:6,background:"rgba(255,255,255,0.08)",borderRadius:3,overflow:"hidden"}}>
                  <div style={{height:"100%",width:`${result.quality*10}%`,borderRadius:3,
                    background:`linear-gradient(90deg,${qualColor}88,${qualColor})`,
                    boxShadow:`0 0 8px ${qualColor}66`,transition:"width 1s ease"}}/>
                </div>
              </div>
            </div>

            {/* Advice */}
            {result.advice && (
              <div style={{
                background:"rgba(255,255,255,0.03)", border:"1px solid rgba(255,255,255,0.08)",
                borderRadius:16, padding:"20px 22px", marginBottom:20
              }}>
                <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:12}}>
                  <span style={{fontSize:16}}>💡</span>
                  <span style={{fontSize:12,fontWeight:700,color:"rgba(255,255,255,0.5)",
                    letterSpacing:0.8,textTransform:"uppercase"}}>Tư vấn từ AI</span>
                </div>
                <p style={{margin:0,color:"rgba(255,255,255,0.75)",fontSize:14,lineHeight:1.7,fontWeight:300}}>{result.advice}</p>
              </div>
            )}

            {/* Input summary */}
            <div style={{background:"rgba(255,255,255,0.02)",border:"1px solid rgba(255,255,255,0.06)",
              borderRadius:14,padding:"14px 18px",marginBottom:20}}>
              <div style={{fontSize:11,fontWeight:700,color:"rgba(255,255,255,0.35)",letterSpacing:0.8,
                textTransform:"uppercase",marginBottom:10}}>Thông tin đã nhập</div>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"6px 20px"}}>
                {[
                  ["Giới tính", vals.gender], ["Tuổi", `${vals.age} tuổi`],
                  ["Giờ ngủ", `${vals.sleep_duration}h`], ["Vận động", `${vals.physical_activity}/100`],
                  ["Stress", `${vals.stress_level}/10`], ["BMI", vals.bmi_category],
                  ["Nhịp tim", `${vals.heart_rate} bpm`], ["Bước chân", vals.daily_steps.toLocaleString()],
                ].map(([k,v])=>(
                  <div key={k} style={{display:"flex",justifyContent:"space-between",fontSize:12}}>
                    <span style={{color:"rgba(255,255,255,0.35)"}}>{k}</span>
                    <span style={{color:"rgba(255,255,255,0.7)",fontWeight:500}}>{v}</span>
                  </div>
                ))}
              </div>
            </div>

            <button onClick={()=>{setStep(0);setResult(null);}} style={{
              width:"100%", padding:"16px",
              background:"rgba(255,255,255,0.05)", border:"1px solid rgba(255,255,255,0.12)",
              borderRadius:14, color:"white", fontSize:14, fontWeight:600,
              cursor:"pointer", fontFamily:"inherit", letterSpacing:0.3,
              transition:"all 0.2s ease"
            }}
              onMouseOver={e=>e.currentTarget.style.background="rgba(255,255,255,0.08)"}
              onMouseOut={e=>e.currentTarget.style.background="rgba(255,255,255,0.05)"}
            >
              ← Phân tích lại
            </button>
          </div>
        ) : null}

        <p style={{textAlign:"center",color:"rgba(255,255,255,0.2)",fontSize:11,marginTop:32,letterSpacing:0.3}}>
          Kết quả chỉ mang tính tham khảo · Không thay thế tư vấn y tế
        </p>
      </div>
    </div>
  );
}
