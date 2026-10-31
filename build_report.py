"""Build numerical tables, the Chinese PDF, and an editable Word document."""
from pathlib import Path
import re
import shutil
import subprocess
import sys
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES, REPORT = ROOT / "results", ROOT / "report"


def mean_sd(values, places=3, scale=1):
    a = np.asarray(values, dtype=float) * scale
    return rf"${a.mean():.{places}f}\pm{a.std(ddof=1):.{places}f}$"


def sci(value):
    mant, exp = f"{value:.2e}".split("e")
    return rf"${mant}\times10^{{{int(exp)}}}$"


def table(name, caption, label, columns, headers, rows, footnote=""):
    text = ("\\begin{table}[htbp]\n\\centering\\small\n"
            + "\\caption{"+caption+"}\\label{"+label+"}\n"
            + "\\begin{tabular}{"+columns+"}\n\\toprule\n"
            + " & ".join(headers)+" \\\\\n\\midrule\n"
            + "\n".join(" & ".join(row)+" \\\\" for row in rows)
            + "\n\\bottomrule\n\\end{tabular}\n")
    if footnote:
        text += "\\par\\vspace{0.3em}{\\footnotesize "+footnote+"}\n"
    text += "\\end{table}\n"
    (REPORT / "tables" / f"{name}.tex").write_text(text)


def write_tables_and_numbers():
    (REPORT / "tables").mkdir(exist_ok=True)
    b = pd.read_csv(RES / "benchmark.csv")
    s = pd.read_csv(RES / "selected_models.csv")
    p = pd.read_csv(RES / "step_sizes.csv")
    refs = pd.read_csv(RES / "reference_checks.csv")
    base = s[s.scenario == "base"]
    meta = json.loads((RES / "metadata.json").read_text())
    values = {}
    for rho, suffix in [(0.,"Independent"),(.8,"Correlated")]:
        a = b[b.rho==rho].groupby("method")[["iterations","seconds"]].mean()
        values["IterSpeed"+suffix] = f"{a.loc['ISTA','iterations']/a.loc['FISTA','iterations']:.2f}"
        values["TimeSpeed"+suffix] = f"{a.loc['ISTA','seconds']/a.loc['FISTA','seconds']:.2f}"
    values.update(BaseMSE=mean_sd(base.test_mse), BaseError=mean_sd(base.relative_error,4),
                  BaseErrorMean=f"{base.relative_error.mean():.4f}",
                  BaseNNZ=f"{base.nnz.mean():.1f}",
                  BasePrecision=f"{base.precision.mean():.3f}",
                  BaseRecall=f"{base.recall.mean():.3f}",
                  BaseNoiseVariance=f"{base.noise_variance.mean():.4f}",
                  BaseNoiseRatio=f"{(base.test_mse/base.noise_variance).mean():.2f}",
                  CPUName=meta["cpu"], PythonVersion=meta["python"].split()[0],
                  NumpyVersion=meta["numpy"], ScipyVersion=meta["scipy"],
                  SpectralMilliseconds=f"{pd.read_csv(RES/'spectral_setup.csv').query('rho==0').L_seconds.mean()*1000:.2f}",
                  MaxObjectiveDifference=sci(max(b.objective_difference.abs().max(),
                    s.objective_difference.abs().max(),s.ista_objective_difference.abs().max())),
                  MaxReferenceResidual=sci(max(refs.relative_gap.max(),refs.relative_kkt.max())),
                  SmallStepIterations=f"{p[p.step_factor==.25].iterations.mean():.1f}",
                  FullStepIterations=f"{p[p.step_factor==1].iterations.mean():.1f}",
                  TotalSeconds=f"{meta['total_seconds']:.1f}")
    example=base[base.seed==20261031].iloc[0]
    for name,col in [("ExampleNNZ","nnz"),("ExampleTP","tp"),("ExampleFP","fp")]:
        values[name]=str(int(example[col]))
    for name,scenario,col in [("ErrorLowSample","n150","relative_error"),
                              ("ErrorHighSample","n600","relative_error"),
                              ("MSELowSample","n150","test_mse"),
                              ("MSEHighSample","n600","test_mse"),
                              ("ErrorLowNoise","noise005","relative_error"),
                              ("ErrorHighNoise","noise030","relative_error"),
                              ("ErrorHighCorrelation","rho090","relative_error")]:
        values[name]=f"{s[s.scenario==scenario][col].mean():.4f}"
    (REPORT / "generated_numbers.tex").write_text(
        "% Generated from actual results; do not hand-edit.\n"+
        "\n".join("\\newcommand{\\"+name+"}{"+value+"}" for name,value in values.items())+"\n")

    rows=[]
    for rho in [0.,.8]:
        for method in ["ISTA","FISTA"]:
            a=b[(b.rho==rho)&(b.method==method)]
            rows.append([f"{rho:g}",method,mean_sd(a.iterations,1),mean_sd(a.seconds,2,1000),
                         sci(a.relative_kkt.max()),sci(a.relative_gap.max())])
    table("benchmark","固定正则化强度下的算法比较（5 个种子）","tab:benchmark","llrrrr",
          [r"$\rho$","方法","迭代次数","时间 / ms",r"最大 $R_{\rm KKT}$",r"最大 $G_{\rm rel}$"],rows)
    rows=[]
    for factor in [.25,.5,1.]:
        a=p[p.step_factor==factor]
        rows.append([rf"${factor:g}/L$",mean_sd(a.iterations,1),sci(a.relative_kkt.max()),sci(a.relative_gap.max())])
    table("steps","ISTA 步长实验（5 个种子）","tab:steps","lrrr",
          ["步长","迭代次数",r"最大 $R_{\rm KKT}$",r"最大 $G_{\rm rel}$"],rows)
    baselines=pd.read_csv(RES/"baselines.csv")
    rows=[]
    for name,cn in [("Lasso (validation)","Lasso（验证集选参）"),
                    ("Minimum-norm LS","最小范数最小二乘"),("Intercept only","仅截距模型")]:
        a=baselines[baselines.model==name]
        rows.append([cn,mean_sd(a.test_mse),mean_sd(a.relative_error,4),mean_sd(a.nnz,1)])
    table("baselines","基础场景下的统计模型比较（5 个种子）","tab:baselines","lrrr",
          ["模型","测试 MSE","参数相对误差","非零数"],rows)
    sweep=pd.read_csv(RES/"lambda_sweep.csv")
    rows=[]
    for ratio,a in sweep[sweep.scenario=="base"].groupby("lambda_ratio"):
        rows.append([f"{ratio:g}",mean_sd(a.test_mse),mean_sd(a.relative_error,4),
                     mean_sd(a.nnz,1),f"{a.precision.mean():.3f}",f"{a.recall.mean():.3f}"])
    table("regularization","基础场景的正则化参数影响","tab:regularization","lrrrrr",
          [r"$\lambda/\lambda_{\max}$","测试 MSE","参数误差","非零数","精确率","召回率"],rows,
          "前 3 项统计量为均值与样本标准差；精确率和召回率列仅列均值。")
    rows=[]
    scenarios=[("base","基础场景"),("n150",r"$n=150$"),("n600",r"$n=600$"),
               ("noise005",r"$\delta=0.05$"),("noise030",r"$\delta=0.3$"),
               ("rho050",r"$\rho=0.5$"),("rho090",r"$\rho=0.9$")]
    for name,label in scenarios:
        a=s[s.scenario==name]
        rows.append([label,mean_sd(a.relative_error,4),mean_sd(a.test_mse),
                     mean_sd(a.nnz,1),mean_sd(a.f1,3)])
    table("sensitivity","验证集选参后的单因素实验（每行 5 个种子）","tab:sensitivity","lrrrr",
          ["场景","参数相对误差","测试 MSE","非零数","支持集 F1"],rows,
          r"基础场景为 $n=300,\ \rho=0,\ \delta=0.1$；每行只改变标出的一个因素。")
    (RES / "report_numbers.json").write_text(json.dumps(values,ensure_ascii=False,indent=2))


def expand_inputs(text, base):
    pattern = re.compile(r"\\input\{([^}]+)\}")
    def replace(m):
        p=base/m.group(1)
        if not p.suffix:
            p=p.with_suffix(".tex")
        return expand_inputs(p.read_text(),p.parent)
    return pattern.sub(replace,text)


def export_docx():
    import pypandoc
    from docx import Document
    from docx.shared import Cm, Pt, RGBColor
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    text=expand_inputs((REPORT/"lasso_report.tex").read_text(),REPORT)
    text=re.sub(r"\{\\rm\s+([^}]+)\}",r"{\\mathrm{\1}}",text)
    # Pandoc reads LaTeX math into editable Word OMML. Use PNG for Word images.
    text=re.sub(r"(figures/[^{}]+)\.pdf",r"\1.png",text)
    # Replace cross-references by the numbers resolved by XeLaTeX.
    aux=(REPORT/"lasso_report.aux").read_text()
    labels=dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]+)\}",aux))
    text=re.sub(r"\\eqref\{([^}]+)\}",lambda m:"("+labels.get(m.group(1),m.group(1))+")",text)
    text=re.sub(r"\\ref\{([^}]+)\}",lambda m:labels.get(m.group(1),m.group(1)),text)
    refs={"tibshirani":"1","daubechies":"2","beck":"3","sklearn":"4"}
    text=re.sub(r"\\cite\{([^}]+)\}",lambda m:"["+refs[m.group(1)]+"]",text)
    # Preserve the PDF's equation numbering in Word. Split the three-line
    # FISTA align environment into individual editable display equations.
    def split_align(m):
        return "\n".join(r"\begin{equation}"+line.replace("&","")+r"\end{equation}"
                         for line in re.split(r"\\\\",m.group(1)) if line.strip())
    text=re.sub(r"\\begin\{align\}(.*?)\\end\{align\}",split_align,text,flags=re.S)
    equation_number=0
    def number_equation(m):
        nonlocal equation_number
        equation_number+=1
        return (r"\begin{equation}"+m.group(1)
                +r"\qquad\text{("+str(equation_number)+r")}\end{equation}")
    text=re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}",number_equation,text,flags=re.S)
    # Make bibliography labels explicit for the editable document.
    text=text.replace(r"\begin{thebibliography}{9}",r"\section*{参考文献}\begin{enumerate}")
    text=text.replace(r"\end{thebibliography}",r"\end{enumerate}")
    text=re.sub(r"\\bibitem\{[^}]+\}",r"\\item ",text)
    # Author styles are handled below; remove PDF-only header/footer commands.
    text=text.replace(r"\FloatBarrier","")
    (REPORT/"word_source.tex").write_text(text)
    output=REPORT/"Lasso研究报告.docx"
    pypandoc.convert_file(str(REPORT/"word_source.tex"),"docx",format="latex",
                         outputfile=str(output),
                         extra_args=["--resource-path="+str(REPORT),"--number-sections"])
    doc=Document(output)
    caption_counts={"Table Caption":0,"Image Caption":0}
    for para in doc.paragraphs:
        if para.text=="Abstract":
            para.text="摘要"
        if para.style.name in caption_counts:
            caption_counts[para.style.name]+=1
            label="表" if para.style.name=="Table Caption" else "图"
            prefix=OxmlElement("w:r")
            t=OxmlElement("w:t")
            t.text=f"{label} {caption_counts[para.style.name]}　"
            prefix.append(t)
            para._p.insert(1 if para._p.pPr is not None else 0,prefix)
    for shape in doc.inline_shapes:
        if shape.width>Cm(16.5):
            ratio=Cm(16.5)/shape.width
            shape.width=Cm(16.5)
            shape.height=int(shape.height*ratio)
    for section in doc.sections:
        section.page_width,section.page_height=Cm(21),Cm(29.7)
        section.top_margin=section.bottom_margin=Cm(2.3)
        section.left_margin=section.right_margin=Cm(2.25)
        footer=section.footer.paragraphs[0]
        footer.alignment=1
        field=OxmlElement("w:fldSimple")
        field.set(qn("w:instr"),"PAGE")
        footer._p.append(field)
    for name in ["Normal","Body Text","First Paragraph"]:
        if name in doc.styles:
            st=doc.styles[name]
            st.font.name="Times New Roman"
            st.font.size=Pt(10.5)
            st._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"),"宋体")
            st.paragraph_format.line_spacing=1.18
            st.paragraph_format.space_after=Pt(4)
    for name in ["Title","Heading 1","Heading 2","Heading 3"]:
        if name in doc.styles:
            st=doc.styles[name]
            st.font.name="Arial"
            st._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"),"黑体")
            st.font.color.rgb=RGBColor.from_string("235789")
    # Keep cells compact and repeat column headings over a page break.
    for tbl in doc.tables:
        for row in tbl.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    for run in para.runs:
                        run.font.size=Pt(9)
        header=OxmlElement("w:tblHeader")
        tbl.rows[0]._tr.get_or_add_trPr().append(header)
    doc.core_properties.title="基于近端梯度法的 Lasso 稀疏回归求解与数值实验"
    doc.core_properties.subject="最优化方法研究报告"
    doc.save(output)
    print("Word:",output)


def main():
    write_tables_and_numbers()
    engine=shutil.which("xelatex")
    if not engine:
        raise SystemExit("XeLaTeX not found; tables generated, PDF build requires TeX Live/CTeX.")
    for _ in range(2):
        log=subprocess.run([engine,"-interaction=nonstopmode","-halt-on-error","lasso_report.tex"],
                           cwd=REPORT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (REPORT/"build.log").write_text(log.stdout)
        if log.returncode:
            print(log.stdout[-7000:])
            raise SystemExit(log.returncode)
    shutil.copyfile(REPORT/"lasso_report.pdf",REPORT/"Lasso研究报告.pdf")
    export_docx()
    print("PDF:",REPORT/"Lasso研究报告.pdf")


if __name__=="__main__":
    main()
