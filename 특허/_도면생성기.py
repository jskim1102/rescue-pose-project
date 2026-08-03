# -*- coding: utf-8 -*-
"""rescue-pose 특허 명세서 슬림본 -> 도면(인라인 SVG)+수식조판 포함 HTML 생성."""
import re, html, pathlib

BASE = "/tmp/claude-1000/-home-kim-3090-dev-harness-claude/d244c521-74d3-4a36-b564-ad418711d8b8/scratchpad"
MD = BASE + "/rescue-pose-명세서-slim.md"
OUT = BASE + "/spec.html"
FONT_REG = "/home/kim_3090/.fonts/NotoSansKR-Regular.otf"
FONT_BOLD = "/home/kim_3090/.fonts/NotoSansKR-Bold.otf"

# ---------- SVG helpers ----------
S = ('font-family:NotoKR;')
def box(x,y,w,h,label,sub="",fill="#fff",rx=6,fs=13,bold=False):
    t=f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="#333" stroke-width="1.5"/>'
    fw='700' if bold else '500'
    lines=label.split("\n")
    n=len(lines)+(1 if sub else 0)
    cy=y+h/2-(n-1)*8+4
    for ln in lines:
        t+=f'<text x="{x+w/2}" y="{cy}" text-anchor="middle" font-size="{fs}" font-weight="{fw}" fill="#111" style="{S}">{html.escape(ln)}</text>'
        cy+=16
    if sub:
        t+=f'<text x="{x+w/2}" y="{cy}" text-anchor="middle" font-size="10.5" fill="#666" style="{S}">{html.escape(sub)}</text>'
    return t
def arrow(x1,y1,x2,y2,dash=False,label=""):
    d=' stroke-dasharray="5 4"' if dash else ''
    t=f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#333" stroke-width="1.5" marker-end="url(#ah)"{d}/>'
    if label:
        mx,my=(x1+x2)/2,(y1+y2)/2-5
        t+=f'<text x="{mx}" y="{my}" text-anchor="middle" font-size="10.5" fill="#444" style="{S}">{html.escape(label)}</text>'
    return t
def txt(x,y,s,fs=11,anc="start",fill="#333",bold=False,italic=False):
    fw='700' if bold else '400'
    fst='italic' if italic else 'normal'
    return f'<text x="{x}" y="{y}" text-anchor="{anc}" font-size="{fs}" font-weight="{fw}" font-style="{fst}" fill="{fill}" style="{S}">{html.escape(s)}</text>'
def svg_open(w,h,cap):
    return (f'<figure class="fig"><svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{min(w,700)}px">'
            f'<defs><marker id="ah" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">'
            f'<path d="M0,0 L7,3 L0,6 Z" fill="#333"/></marker></defs>')
def svg_close(cap):
    return f'</svg><figcaption>{html.escape(cap)}</figcaption></figure>'

# subscript formula helper for SVG (tspan)
def fsub(x,y,parts,fs=13,anc="start"):
    # parts: list of (text, is_sub)
    s=f'<text x="{x}" y="{y}" text-anchor="{anc}" font-size="{fs}" fill="#111" style="{S}">'
    for tt,sub in parts:
        if sub: s+=f'<tspan baseline-shift="sub" font-size="{int(fs*0.72)}">{html.escape(tt)}</tspan>'
        else: s+=f'<tspan>{html.escape(tt)}</tspan>'
    return s+'</text>'

# ---------- 도 1 : 전체 시스템 블록도 ----------
def fig1():
    w,h=700,430; s=svg_open(w,h,"")
    # embedded board dashed enclosure
    s+=f'<rect x="24" y="70" width="500" height="330" rx="10" fill="#f7f9fc" stroke="#8aa" stroke-width="1.3" stroke-dasharray="7 5"/>'
    s+=txt(30,90,"임베디드 엣지 연산 보드 (190)",11,"start","#688",bold=True)
    s+=box(40,110,120,54,"제1 영상 획득부","(110) RGB 카메라")
    s+=box(40,190,120,54,"제2 영상 획득부","(120) 적외선 카메라")
    # single NN big box
    s+=f'<rect x="210" y="96" width="150" height="230" rx="8" fill="#eef3fb" stroke="#345" stroke-width="1.6"/>'
    s+=txt(285,114,"단일 신경망 (130)",11.5,"middle","#234",bold=True)
    s+=box(222,124,126,30,"융합부 (133)",fill="#fff",rx=5,fs=11)
    s+=box(222,160,126,26,"조도적응 게이트(134)",fill="#fff",rx=5,fs=10)
    s+=box(222,192,126,26,"조도지표 산출(135)",fill="#fff",rx=5,fs=10)
    s+=box(222,232,126,34,"키포인트 디코더","(140)",fill="#fff",rx=5,fs=11)
    s+=txt(285,300,"→ 골격 키포인트",10.5,"middle","#456")
    s+=box(400,110,110,48,"자세 분류부","(150)")
    s+=box(400,178,110,48,"구조 판단부","(160) 지속·정지")
    s+=box(400,246,110,48,"추적부 (170)","다인 per-person",fs=11)
    s+=box(560,178,116,48,"경보·통신부","(180)")
    s+=box(560,258,116,44,"관제센터/","관리자 단말",fill="#f4f4f4")
    # arrows
    s+=arrow(160,137,210,150)      # rgb->nn
    s+=arrow(160,217,210,205)      # ir->nn
    s+=arrow(360,200,400,150,label="F_fuse→키포인트".replace("F_fuse",""))
    s+=arrow(360,240,400,200)
    s+=arrow(455,158,455,178)      # 150->160
    s+=arrow(455,270,455,226)      # 170->160
    s+=arrow(510,202,560,202,label="구조필요")
    s+=arrow(618,226,618,258)      # 180->관제
    s+=fsub(212,352,[("융합 피처맵  F",False),("fuse",True),(" = w",False),("rgb",True),("·F",False),("rgb",True),(" + w",False),("ir",True),("·F",False),("ir",True)],fs=12)
    return s+svg_close("도 1  응급 상황 감지 시스템(100) 전체 블록도 — 모든 연산이 임베디드 보드(190)에서 수행")

# ---------- 도 2 : 신경망 아키텍처 ----------
def fig2():
    w,h=700,340; s=svg_open(w,h,"")
    s+=box(30,60,96,44,"RGB 영상","[3,H,W]",fill="#fff")
    s+=box(30,200,96,44,"적외선 영상","[1,H,W]",fill="#fff")
    s+=box(165,60,120,44,"RGB 브랜치","(131)",fill="#eef3fb")
    s+=box(165,200,120,44,"적외선 브랜치","(132)",fill="#eef3fb")
    s+=fsub(300,88,[("F",False),("rgb",True)],fs=14)
    s+=fsub(300,228,[("F",False),("ir",True)],fs=14)
    s+=box(340,120,130,64,"융합부 (133)","중간계층 stride-16",fill="#dfeaf7",fs=12)
    s+=txt(405,205,"딥 피처맵 레벨",10.5,"middle","#a33",bold=True)
    s+=txt(405,220,"(입력/픽셀 융합 아님)",10,"middle","#a33")
    s+=fsub(500,150,[("F",False),("fuse",True)],fs=15)
    s+=box(548,118,128,68,"키포인트 디코더","(140) deconv 업샘플",fill="#eef3fb",fs=11.5)
    s+=box(548,214,128,50,"COCO-17","17채널 히트맵",fill="#fff",fs=11)
    s+=arrow(126,82,165,82); s+=arrow(126,222,165,222)
    s+=arrow(285,82,340,140); s+=arrow(285,222,340,164)
    s+=arrow(470,152,548,152)
    s+=arrow(612,186,612,214)
    s+=txt(612,290,"→ 골격 키포인트 단일 세트",10.5,"middle","#456")
    s+=txt(350,315,"두 모달을 특징단계에서 하나의 융합맵으로 결합 → 그 융합맵으로부터 직접 키포인트 디코딩(모달별 결합 아님)",10.5,"middle","#555")
    return s+svg_close("도 2  단일 신경망(130) 아키텍처 — 융합 피처맵으로부터 직접 키포인트 디코딩(브릿지)")

# ---------- 도 3 : soft 조도적응 융합 게이트 ----------
def fig3():
    w,h=680,300; s=svg_open(w,h,"")
    s+=box(30,120,120,50,"조도지표 산출","(135)  g",fill="#fff")
    s+=box(210,110,130,70,"조도적응 게이트","(134)",fill="#dfeaf7",fs=12)
    s+=arrow(150,145,210,145,label="g")
    # weights bars with lower bound
    s+=txt(410,70,"가중치 (연속 가변, 하한 ε>0 클램프)",11,"middle","#234",bold=True)
    for i,(lab,val) in enumerate([("w_rgb",0.72),("w_ir",0.85)]):
        by=100+i*70; bx=400; bw=200
        s+=f'<rect x="{bx}" y="{by}" width="{bw}" height="20" rx="4" fill="#eee" stroke="#bbb"/>'
        s+=f'<rect x="{bx}" y="{by}" width="{int(bw*val)}" height="20" rx="4" fill="#7fa8d8"/>'
        # lower bound marker
        s+=f'<line x1="{bx+int(bw*0.14)}" y1="{by-6}" x2="{bx+int(bw*0.14)}" y2="{by+26}" stroke="#c33" stroke-width="1.5" stroke-dasharray="3 2"/>'
        sub=lab.split("_")[1]
        s+=fsub(bx-52,by+15,[("w",False),(sub,True)],fs=13)
    s+=txt(400+int(200*0.14),210,"ε (하한)",9.5,"middle","#c33")
    s+=arrow(340,145,400,120); s+=arrow(340,150,400,190)
    s+=fsub(150,255,[("F",False),("fuse",True),(" = w",False),("rgb",True),("·F",False),("rgb",True),(" + w",False),("ir",True),("·F",False),("ir",True),("   (두 모달 상시 기여, 카메라 택일 스위칭 아님)",False)],fs=13)
    return s+svg_close("도 3  조도적응 소프트 융합 게이트(134) — 조도에 따라 가중 연속 가변, 각 기여 0<ε 하한 유지")

# ---------- 도 4 : 방법 플로우차트 ----------
def fig4():
    w,h=430,540; s=svg_open(w,h,"")
    steps=[("S210","RGB·적외선 영상 획득·정합"),("S220","딥 피처맵 융합 → 융합맵"),
           ("S230","융합맵→키포인트 직접 디코딩"),("S240","자세(눕기/앉기/서기) 판정")]
    y=30
    for code,lab in steps:
        s+=box(120,y,190,44,lab,code,fill="#eef3fb",fs=11.5)
        s+=arrow(215,y+44,215,y+64) if code!="S240" else ""
        y+=64
    # decision diamond
    dy=y+6
    s+=f'<polygon points="215,{dy} 330,{dy+45} 215,{dy+90} 100,{dy+45}" fill="#fdf5e6" stroke="#333" stroke-width="1.5"/>'
    s+=txt(215,dy+38,"눕기 N초 연속 지속",10.5,"middle","#111",bold=True)
    s+=txt(215,dy+54,"& 정지조건(≤τ)?",10.5,"middle","#111")
    s+=arrow(215,y+44,215,dy)
    # yes -> S250 -> S260
    yy=dy+110
    s+=arrow(215,dy+90,215,yy,label="예")
    s+=box(120,yy,190,44,"구조 필요 판정 (S250)","가속도·충격 미사용",fill="#fde8e8",fs=11.5)
    s+=arrow(215,yy+44,215,yy+64)
    s+=box(120,yy+64,190,40,"경보 전송 (S260)",fill="#fff",fs=11.5)
    # no -> loop back
    s+=arrow(330,dy+45,380,dy+45); s+=f'<line x1="380" y1="{dy+45}" x2="380" y2="20" stroke="#333" stroke-width="1.5"/><line x1="380" y1="20" x2="215" y2="20" stroke="#333" stroke-width="1.5"/>'
    s+=arrow(215,20,215,30,label="아니오")
    # recover cancel note
    s+=txt(20,dy+150,"※ N초 경과 전 앉기/서기 전이 시",10,"start","#a60")
    s+=txt(20,dy+164,"   시간계수 초기화 (회복 취소)",10,"start","#a60")
    return s+svg_close("도 4  방법 흐름도(S210~S260) — 지속·정지 기반 판정 및 회복 취소")

# ---------- 도 5 : 자세 기하 ----------
def stick(cx,cy,pose):
    # returns stick figure svg; pose in {stand,sit,lie}
    c="#333"
    def L(x1,y1,x2,y2): return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="2.4" stroke-linecap="round"/>'
    def H(x,y): return f'<circle cx="{x}" cy="{y}" r="9" fill="none" stroke="{c}" stroke-width="2.4"/>'
    if pose=="stand":
        s=H(cx,cy-46)+L(cx,cy-37,cx,cy)+L(cx,cy-28,cx-16,cy-8)+L(cx,cy-28,cx+16,cy-8)+L(cx,cy,cx-11,cy+34)+L(cx,cy,cx+11,cy+34)
        s+=f'<line x1="{cx}" y1="{cy-37}" x2="{cx}" y2="{cy}" stroke="#c33" stroke-width="1" stroke-dasharray="3 2"/>'
        s+=txt(cx+6,cy-16,"φ≈90°",9.5,"start","#c33")
    elif pose=="sit":
        s=H(cx,cy-30)+L(cx,cy-21,cx,cy+4)+L(cx,cy-14,cx-15,cy-2)+L(cx,cy-14,cx+15,cy-2)+L(cx,cy+4,cx+20,cy+4)+L(cx+20,cy+4,cx+20,cy+30)+L(cx,cy+4,cx-2,cy+30)
        s+=txt(cx+8,cy+20,"무릎각 α",9.5,"start","#369")
    else: # lie
        base=cy+18
        s=H(cx-40,base)+L(cx-31,base,cx+8,base)+L(cx-24,base,cx-30,base-14)+L(cx-24,base,cx-30,base+14)+L(cx+8,base,cx+34,base-6)+L(cx+8,base,cx+34,base+8)
        s+=f'<line x1="{cx-31}" y1="{base}" x2="{cx+8}" y2="{base}" stroke="#c33" stroke-width="1" stroke-dasharray="3 2"/>'
        s+=f'<line x1="{cx-45}" y1="{base+22}" x2="{cx+40}" y2="{base+22}" stroke="#888" stroke-width="1.4"/>'
        s+=txt(cx,base-20,"φ≈0° (|φ|<30°)",9.5,"middle","#c33")
    return s
def fig5():
    w,h=680,240; s=svg_open(w,h,"")
    for i,(pose,lab) in enumerate([("stand","서기"),("sit","앉기"),("lie","눕기")]):
        cx=150+i*200
        s+=stick(cx,110,pose)
        s+=box(cx-70,175,140,34,lab if pose!="lie" else "눕기 → 응급 후보",fill="#fff" if pose!="lie" else "#fde8e8",fs=12,bold=(pose=="lie"))
    s+=f'<line x1="60" y1="150" x2="620" y2="150" stroke="#ccc" stroke-width="1" stroke-dasharray="2 3"/>'
    s+=txt(340,232,"몸통축(어깨중점–엉덩이중점)의 지면 기울기 φ 및 관절각으로 판정 — 경계상자 종횡비 아님",10.5,"middle","#555")
    return s+svg_close("도 5  자세 기하 — 몸통축 지면 기울기 φ 및 관절각에 의한 눕기/앉기/서기 구분")

# ---------- 도 6 : 지속+정지 타이밍도 ----------
def fig6():
    w,h=700,320; s=svg_open(w,h,"")
    x0,x1=60,650;
    # posture track
    s+=txt(20,60,"자세",10.5,"start","#333",bold=True)
    s+=f'<line x1="{x0}" y1="70" x2="{x1}" y2="70" stroke="#ccc"/>'
    s+=f'<rect x="{x0}" y="52" width="120" height="18" fill="#dfeaf7"/><text x="{x0+60}" y="66" text-anchor="middle" font-size="10" style="{S}">서기</text>'
    s+=f'<rect x="{x0+120}" y="52" width="360" height="18" fill="#fde8e8"/><text x="{x0+300}" y="66" text-anchor="middle" font-size="10" style="{S}">눕기 (연속)</text>'
    s+=f'<rect x="{x0+480}" y="52" width="110" height="18" fill="#dfeaf7"/><text x="{x0+535}" y="66" text-anchor="middle" font-size="10" style="{S}">앉기</text>'
    # timer ramp
    s+=txt(20,150,"지속 타이머",10.5,"start","#333",bold=True)
    s+=f'<line x1="{x0}" y1="200" x2="{x1}" y2="200" stroke="#ccc"/>'
    s+=f'<polyline points="{x0+120},200 {x0+360},120 {x0+360},120" fill="none" stroke="#c33" stroke-width="2"/>'
    s+=f'<line x1="{x0+360}" y1="120" x2="{x0+480}" y2="120" stroke="#c33" stroke-width="2"/>'
    s+=f'<line x1="{x0+480}" y1="120" x2="{x0+480}" y2="200" stroke="#c33" stroke-width="2" stroke-dasharray="4 3"/>'
    s+=f'<line x1="{x0}" y1="128" x2="{x1}" y2="128" stroke="#999" stroke-width="1" stroke-dasharray="5 4"/>'
    s+=txt(x1,124,"N초 임계",9.5,"end","#666")
    s+=f'<circle cx="{x0+360}" cy="120" r="5" fill="#c33"/>'
    s+=txt(x0+360,108,"구조필요 트리거 (N초 도달 & 정지 ≤τ)",9.5,"middle","#c33",bold=True)
    s+=txt(x0+490,150,"앉기 전이→",9.5,"start","#a60")
    s+=txt(x0+490,164,"계수 초기화",9.5,"start","#a60")
    s+=txt(x0+490,178,"(회복취소)",9.5,"start","#a60")
    s+=f'<line x1="{x0}" y1="230" x2="{x1}" y2="230" stroke="#333" marker-end="url(#ah)"/>'
    s+=txt(x1,248,"시간",10,"end","#333")
    return s+svg_close("도 6  눕기 지속(N초)+정지(≤τ) 판정 타이밍도 및 회복 취소")

# ---------- 도 7 : 다인 추적/보간 ----------
def fig7():
    w,h=680,290; s=svg_open(w,h,"")
    s+=txt(30,30,"프레임 →",11,"start","#333",bold=True)
    cols=[90,230,370,510]
    for i,fx in enumerate(cols):
        s+=f'<rect x="{fx}" y="45" width="120" height="180" rx="6" fill="#fafbfe" stroke="#ccd"/>'
        s+=txt(fx+60,40,f"t{i+1}",10,"middle","#888")
    # ID1 present all, ID2 missing at t3 (interpolate)
    def person(fx,fy,col,mid=False):
        c=col
        da=' stroke-dasharray="3 2"' if mid else ''
        s2=f'<circle cx="{fx}" cy="{fy}" r="7" fill="none" stroke="{c}" stroke-width="2"{da}/>'
        s2+=f'<line x1="{fx}" y1="{fy+7}" x2="{fx}" y2="{fy+34}" stroke="{c}" stroke-width="2"{da}/>'
        s2+=f'<line x1="{fx}" y1="{fy+16}" x2="{fx-12}" y2="{fy+28}" stroke="{c}" stroke-width="2"/><line x1="{fx}" y1="{fy+16}" x2="{fx+12}" y2="{fy+28}" stroke="{c}" stroke-width="2"/>'
        return s2
    for i,fx in enumerate(cols):
        s+=person(fx+35,80,"#2a6")   # ID1
        s+=person(fx+80,120,"#c60", mid=(i==2))  # ID2, missing at t3
    s+=txt(cols[0]+35,230,"ID1",9.5,"middle","#2a6",bold=True)
    s+=txt(cols[0]+80,250,"ID2",9.5,"middle","#c60",bold=True)
    s+=txt(cols[2]+80,245,"미검출→보간",9,"middle","#c60")
    s+=txt(340,282,"각 사람 키포인트로 식별·추적(ID), 사람별 독립 지속·정지 타이머, 미검출 프레임은 이전·이후 키포인트로 보간",10,"middle","#555")
    return s+svg_close("도 7  다인 추적 및 미검출 프레임 키포인트 보간 — 사람별(per-person) 판정")

FIGS={1:fig1,2:fig2,3:fig3,4:fig4,5:fig5,6:fig6,7:fig7}

# ---------- minimal markdown -> html ----------
def inline(t):
    t=html.escape(t)
    t=re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', t)
    # subscripts for tokens like F_fuse, w_rgb, I_rgb
    t=re.sub(r'\b([FwI])_(fuse|rgb|ir)\b', r'\1<sub>\2</sub>', t)
    return t

def md_to_html(md):
    out=[]; i=0; lines=md.split("\n"); n=len(lines)
    while i<n:
        ln=lines[i]
        if not ln.strip():
            i+=1; continue
        m=re.match(r'^(#{1,4})\s+(.*)$', ln)
        if m:
            lvl=len(m.group(1)); out.append(f'<h{lvl}>{inline(m.group(2))}</h{lvl}>'); i+=1; continue
        # table
        if ln.lstrip().startswith('|') and i+1<n and re.match(r'^\s*\|[\s:\-|]+\|\s*$', lines[i+1]):
            hdr=[c.strip() for c in ln.strip().strip('|').split('|')]
            out.append('<table><thead><tr>'+''.join(f'<th>{inline(c)}</th>' for c in hdr)+'</tr></thead><tbody>')
            i+=2
            while i<n and lines[i].lstrip().startswith('|'):
                cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
                out.append('<tr>'+''.join(f'<td>{inline(c)}</td>' for c in cells)+'</tr>'); i+=1
            out.append('</tbody></table>'); continue
        # paragraph (gather until blank)
        buf=[ln]; i+=1
        while i<n and lines[i].strip() and not re.match(r'^(#{1,4})\s', lines[i]) and not lines[i].lstrip().startswith('|'):
            buf.append(lines[i]); i+=1
        para=' '.join(buf)
        out.append(f'<p>{inline(para)}</p>')
        # inject figure right after a 도면 간단설명 paragraph "도 N..."
        mfig=re.match(r'^도\s*([1-7])[은는]', para)
        if mfig:
            out.append(FIGS[int(mfig.group(1))]())
    return "\n".join(out)

md=pathlib.Path(MD).read_text(encoding="utf-8")
body=md_to_html(md)

CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
@font-face{font-family:'NotoKR';src:url('file://__FONTREG__') format('opentype');font-weight:400;}
@font-face{font-family:'NotoKR';src:url('file://__FONTBOLD__') format('opentype');font-weight:700;}
*{box-sizing:border-box;}
html,body{font-family:'NotoKR',sans-serif;color:#161616;font-size:10.6pt;line-height:1.62;margin:0;}
h1{font-size:16pt;text-align:center;margin:0 0 6pt;padding-bottom:6pt;border-bottom:2px solid #333;}
h2{font-size:12.5pt;margin:16pt 0 5pt;padding:3pt 0 3pt 8pt;border-left:4px solid #4a6da7;background:#f2f6fc;}
h3{font-size:11.5pt;margin:12pt 0 4pt;color:#234;font-weight:700;}
h4{font-size:10.8pt;margin:9pt 0 3pt;color:#345;font-weight:700;}
p{margin:0 0 6pt;text-align:justify;}
strong{font-weight:700;}
sub{font-size:.72em;}
table{border-collapse:collapse;width:100%;margin:6pt 0;font-size:9.6pt;}
th,td{border:1px solid #bbb;padding:3pt 5pt;text-align:left;vertical-align:top;}
th{background:#eef2f8;}
figure.fig{margin:10pt auto;text-align:center;page-break-inside:avoid;border:1px solid #e2e6ee;border-radius:8px;padding:10pt 8pt 6pt;background:#fff;}
figure.fig svg{display:block;margin:0 auto;}
figcaption{font-size:9.4pt;color:#555;margin-top:6pt;font-weight:500;}
h2,h3{page-break-after:avoid;}
""".replace("__FONTREG__", FONT_REG).replace("__FONTBOLD__", FONT_BOLD)

htmlout=f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><style>{CSS}</style></head><body>{body}</body></html>"""
pathlib.Path(OUT).write_text(htmlout, encoding="utf-8")
print("HTML written:", OUT, len(htmlout), "chars; figures:", sum(1 for _ in re.finditer('figure class="fig"', htmlout)))
