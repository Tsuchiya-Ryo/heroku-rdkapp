from flask import Flask, render_template, request, redirect, url_for, send_from_directory, send_file
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql.expression import func, select
import os, base64, shutil
from rdkit import Chem
import numpy as np
import pubchempy as pcp
from my_modules import random_string, label, calc_specs, calc_gasteiger, type_mapping, charge_value_mapping,\
     label_mapping, standardize, smiles_or_inchi, calc_bits, similarity_map, grid_image
from datetime import datetime

#--------------------------------------------------------------------------------------------------------------------------------
import io
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_dangerously_set_inner_html as dhtml
import dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
 
import argparse
from rdkit.Chem import Descriptors, AllChem, DataStructs, Draw
from rdkit.Chem.Draw import rdDepictor
import plotly.graph_objs as go

#---------------------------------------------------------------------------------------------------------------------------------

app = Flask(__name__, static_folder="tempimage")
app.config["SQLALCHEMY_DATABASE_URI"] = "<YOUR_DATABASE_URI>"
#APP_ROOT = os.path.dirname(os.path.abspath(__file__))
db = SQLAlchemy(app)

#---------------------------------------------------------------------------------------------------------------------------------
descdict = dict(Descriptors._descList)
DisplayItems = ["MolWt","HeavyAtomMolWt","ExactMolWt","HeavyAtomCount","FractionCSP3","NumHAcceptors",\
    "NumHDonors","NumHeteroatoms","NumRotatableBonds","NumAromaticRings","MolLogP","TPSA"]
descdict = {k: v for k, v in descdict.items() if k in DisplayItems}
 
app_dash = dash.Dash(__name__, server=app, url_base_pathname="/sdfplotly/", external_stylesheets=[dbc.themes.BOOTSTRAP])
 
css_directory = os.getcwd()+'/static/'
stylesheets = ['bWLwgP.css']
static_css_route = '/static/'
 
def smi2svg(smi):
    mol = Chem.MolFromSmiles(smi)
    rdDepictor.Compute2DCoords(mol)
    mc = Chem.Mol(mol.ToBinary())
    Chem.Kekulize(mc)
    drawer = Draw.MolDraw2DSVG(300,300)
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:','')
    return svg
 
def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    with open('./static/temp.sdf', 'w') as f:
        f.write(decoded.decode('utf-8'))
    mols = [mol for mol in Chem.SDMolSupplier('./static/temp.sdf')]
    return mols
 
app_dash.layout = dbc.Container(children=[

    dbc.Navbar(dark=True, color="dark",
    children=[dbc.NavbarBrand(html.A("Home", href="/"), className="ml-2"),
                          dbc.NavLink(html.A("Database", href="/explore")),
                          dbc.NavLink(html.A("Check", href="/check")),
                          dbc.NavLink(html.A("Similarity", href="/similarity")),
                          dbc.NavLink(html.A("About", href="/about"))
    ],),

# ),
    html.Br(),
    html.H1(children='Summerize molecules'),
    html.A("ref of this plot function", href="https://iwatobipen.wordpress.com/2019/02/16/make-interactive-dashboard-with-dash2-chemoinformatcs-rdkit/", target="_blank"),
    dcc.Upload(
        id='upload-data',
        children=html.Div(
            html.A('Upload sdf file')
            ),
            style={
                    'width': '40%',
                    'height': '30px',
                    'lineHeight': '30px',
                    'borderWidth': '1px',
                    'borderStyle': 'solid',
                    'borderRadius': '9px',
                    'textAlign': 'center',
                    'margin': '30px',
                    "backgroundcolor":"blue"
            },
            multiple=True
        ),
     
    html.Div([dcc.Dropdown(id='x-column',
                           value='MolWt',
                           options=[{'label': key, 'value': key} for key in descdict.keys()],
                           style={'width':'50%', 'display':'inline-block'}),
              dcc.Dropdown(id='y-column',
                           value='MolLogP',
                           options=[{'label': key, 'value': key} for key in descdict.keys()],
                           style={'width':'50%', 'display':'inline-block'}),
                           ]),
    html.Div([
        html.Div([dcc.Graph(id='example-graph')], className="eight columns")
        ], className="row"),
        html.Div([html.Div(id="molimg")], className="four columns",
        style={"padding-bottom":"500px"}),
    ])
 
@app_dash.callback(
    Output('example-graph', 'figure'),
      
    [Input('upload-data', 'contents'),
     Input('x-column', 'value'),
     Input('y-column', 'value')]
)
def update_graph(contents, x_column_name, y_column_name):
    mols = parse_contents(contents[0])
    for i, mol in enumerate(mols):
        AllChem.Compute2DCoords(mol)
    x = [descdict[x_column_name](mol) for mol in mols]
    y = [descdict[y_column_name](mol) for mol in mols]

    return {'data':[go.Scatter(
        x=x,
        y=y,
        text=[Chem.MolToSmiles(mol) for mol in mols],
        mode='markers',
        marker={
            'size':12,
            'opacity':0.5
        }
    )],
    'layout':go.Layout(
        xaxis={'title':x_column_name},
        yaxis={'title':y_column_name}
    )}
 
 
@app_dash.callback(
    Output('molimg', 'children'),
    [Input('example-graph', 'hoverData'),
    ]
)
def update_img(hoverData1):
    try:
        svg = smi2svg(hoverData1['points'][0]['text'])
        plotdata =  "({},{}) \n {}".format(round(hoverData1["points"][0]["x"],3), round(hoverData1["points"][0]["y"],3),hoverData1["points"][0]["text"])
        # plotdata = plotdata
        return dhtml.DangerouslySetInnerHTML(svg), plotdata
    except:
        svg = 'Select Datapoint'
        return dhtml.DangerouslySetInnerHTML(svg)
 
#--------------------------------------------------------------------------------------------------------------------------------


############# DEFINE DATABASE ########################################################

class ChemData(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    Cid = db.Column(db.String(25), nullable=False, default="N/A")
    Smiles = db.Column(db.String(300), nullable=False, default="N/A")
    MolwtSC = db.Column(db.FLOAT, nullable=False, default=0)
    Fbond = db.Column(db.FLOAT, nullable=False, default=0)
    Frbond = db.Column(db.FLOAT, nullable=False, default=0)
    Faratom = db.Column(db.FLOAT, nullable=False, default=0)
    Fsp3carb = db.Column(db.FLOAT, nullable=False, default=0)
    Fhetero = db.Column(db.FLOAT, nullable=False, default=0)
    Molwt = db.Column(db.FLOAT, nullable=False, default=0)
    Singlebond = db.Column(db.FLOAT, nullable=False, default=0)
    Doublebond = db.Column(db.FLOAT, nullable=False, default=0)
    Triplebond = db.Column(db.FLOAT, nullable=False, default=0)
    Rotbond = db.Column(db.FLOAT, nullable=False, default=0)
    Aratom = db.Column(db.FLOAT, nullable=False, default=0)
    Heteroatom = db.Column(db.FLOAT, nullable=False, default=0)
    Impath = db.Column(db.String(200), nullable=False, default="N/A")

    def __repr__(self):
        return "ChemData " + str(self.id)

############# HOME PAGE ########################################################

@app.route("/")
def index():
    return render_template('home.html')

############# EXPLORE DATABASE ########################################################
    
@app.route("/explore", methods=["GET", "POST"])
def posts():
    if request.method == "POST":
        r1, r2, r3, r4, r5, r6 = ChemData.MolwtSC, ChemData.Fbond, ChemData.Frbond, ChemData.Faratom, ChemData.Fsp3carb, ChemData.Fhetero
        molorder = request.form["radio1"]
        bondorder = request.form["radio2"]
        frameorder = request.form["radio3"]
        aromaorder = request.form["radio4"]
        aliphaorder = request.form["radio5"]
        heteroorder = request.form["radio6"]
        comment = {}
        comment["MolWt"], comment["Bonds"], comment["Frame"], comment["Aromaticity"], comment["Aliphaticity"], comment["Hetero"] = \
            molorder, bondorder, frameorder, aromaorder, aliphaorder, heteroorder
        
        all_posts = (ChemData.query
                     .filter(r1>=0.6 if molorder=="Large" else r1<=0.6, r1>=-0.112 if molorder=="Medium" else r1<=-0.112 if molorder=="Small" else True)
                     .filter(r2>=0.259 if bondorder=="Thick" else r2<=0.259 if bondorder=="Thin" else True)
                     .filter(r3<=0.166 if frameorder=="Hard" else r3>=0.166 if frameorder=="Soft" else True)
                     .filter(r4>=0.428 if aromaorder=="Strong" else r4<=0.428 if aromaorder=="Weak" else True)
                     .filter(r5>=0.368 if aliphaorder=="Strong" else r5<=0.368 if aliphaorder=="Weak" else True)
                     .filter(r6>=0.266 if heteroorder=="More" else r6<=0.266 if heteroorder=="Less" else True)
                    ).order_by(func.random()).first()   
        #all_posts = ChemData.query.filter(r1 >= 360, r2 >= 0.275, r3 >= 4, r4 >= 0.428, r5 >= 0.368, r6 >= 0.275).order_by(func.random()).first()
        #image = base64.b64encode(all_posts.IMG).decode("ascii")
        if os.path.exists("./tempimage"):
            shutil.rmtree("./tempimage")

        img_dir = "./tempimage"
        if not os.path.isdir(img_dir):
            os.mkdir(img_dir)
        all_posts.Molwt = round(all_posts.Molwt, 3)
        mol = Chem.MolFromSmiles(all_posts.Smiles)
        _, img, filename, save_path = calc_specs(mol, img_dir, labelnum_on_structure=False)
        with open(save_path, "w") as f:
            f.write(img)
            f.close()
        # img.save(save_path)
        return render_template("explore.html", posts=all_posts, image=filename, comment=comment)
    else:
        return render_template("index.html")

############# CHECK MOLECULE BY SMILES AND INCHI ########################################################

@app.route("/check", methods=["GET", "POST"])
def check():
    if request.method == "POST":
        if not request.form["smiles"]:
            return render_template("check.html", message=None)

        smiles = request.form["smiles"]

        if Chem.MolFromSmiles(smiles):
            mol = Chem.MolFromSmiles(smiles)
        elif Chem.MolFromInchi(smiles):
            mol = Chem.MolFromInchi(smiles)
        else:
            mol = None

        comment = {}
        comment["Inputted Molecule"] = smiles
        
        if mol is None:
            generate_message = "Failed to generate mol object."
            return render_template("check.html", message=generate_message)
        
        if request.form.get("standardize") == "On":
            mol = standardize(mol)
            comment["Condition"] = "desalted and neutralized"
        
        if mol is None:
            standardize_message = "Failed to desalt."
            return render_template("check.html", message=standardize_message)

        if os.path.exists("./tempimage"):
            shutil.rmtree("./tempimage")

        img_dir = "./tempimage"
        if not os.path.isdir(img_dir):
            os.mkdir(img_dir)

        # CALC MOLECULE SPECIFICATION AND DRAWING
        spec, img, filename, save_path = calc_specs(mol, img_dir)
        with open(save_path, "w") as f:
            f.write(img)
            f.close()
        # img.save(save_path)

        # ATOM LABEL MAPPING
        # label_img, label_filename, label_save_path = label_mapping(mol, img_dir)
        # label_img.save(label_save_path)

        # ATOM TYPE MAPPING
        _, _, _, types_dict = type_mapping(mol, img_dir)
        # type_img.save(type_save_path)

        # if request.form.get("as_table") == "On":
        #     # GASTEIGER CHARGE MAPPING
        #     if request.form.get("gasteiger") == "On":
        #         gmol, charges, gimg, gfilename, gsave_path = calc_gasteiger(mol, img_dir)

        #         # CHARGE VALUE MAPPING
        #         # ndigits = int(request.form["digits"])
        #         # charge_img, charge_filename, charge_save_path, charge_dict = charge_value_mapping(gmol, charges, ndigits, img_dir)           
        #         # gimg.savefig(gsave_path, bbox_inches="tight")
        #         # charge_img.save(charge_save_path)

        #         return render_template("check_result_gasteiger.html", image=filename, gimage=gfilename, comment=comment, spec=spec)

        #     else:
        #         return render_template("check_result.html", image=filename, comment=comment, spec=spec)
        

            # GASTEIGER CHARGE MAPPING
        if request.form.get("gasteiger") == "On":
            gmol, charges, gimg, gfilename, gsave_path = calc_gasteiger(mol, img_dir)

            # CHARGE VALUE MAPPING
            ndigits = int(request.form["digits"])
            _, _, _, charge_dict = charge_value_mapping(gmol, charges, ndigits, img_dir)

            gimg.savefig(gsave_path, bbox_inches="tight")
            # charge_img.save(charge_save_path)

            return render_template("check_result_gasteiger.html", image=filename, gimage=gfilename, comment=comment, spec=spec, types_dict=types_dict, charge_dict=charge_dict)

        else:
            return render_template("check_result.html", image=filename, comment=comment, spec=spec, types_dict=types_dict)
    
    else:
        return render_template("check.html", message=None)

############# CHECK MOLECULE BY NAME ########################################################

@app.route("/checkname", methods=["GET", "POST"])
def checkname():
    if request.method == "POST":
        if not request.form["name"]:
            return render_template("check.html", message=None)

        name = request.form["name"]
        mol = pcp.get_compounds(name, "name")

        if not mol:       #detect empty list(`pcp.get_compounds` returns list)
            find_message = "Pubchem API couldn't find '{}'.".format(name)
            return render_template("check.html", message=find_message)

        mol = mol[0]
        comment = {}
        comment["Inputted Molecule"] = name

        if mol is None:
            generate_message = "Failed to generate mol object."
            return render_template("check.html", message=generate_message)
        
        smiles = mol.canonical_smiles
        mol = Chem.MolFromSmiles(smiles)
        comment["Inputted SMILES"] = smiles

        if request.form.get("standardize") == "On":
            mol = standardize(mol)
            comment["Condition"] = "desalted and neutralized"

        if mol is None:
            standardize_message = "Failed to desalt."
            return render_template("check.html", message=standardize_message)        

        if os.path.exists("./tempimage"):
            shutil.rmtree("./tempimage")

        img_dir = "./tempimage"
        if not os.path.isdir(img_dir):
            os.mkdir(img_dir)

        # CALC MOLECULE SPECIFICATION AND DRAWING
        spec, img, filename, save_path = calc_specs(mol, img_dir)
        with open(save_path, "w") as f:
            f.write(img)
            f.close()
        # img.save(save_path)

        # ATOM LABEL MAPPING
        # label_img, label_filename, label_save_path = label_mapping(mol, img_dir)
        # label_img.save(label_save_path)

        # ATOM TYPE MAPPING
        _, _, _, types_dict = type_mapping(mol, img_dir)
        # type_img.save(type_save_path)

        # if request.form.get("as_table") == "On":
        #     # GASTEIGER CHARGE MAPPING
        #     if request.form.get("gasteiger") == "On":
        #         gmol, charges, gimg, gfilename, gsave_path = calc_gasteiger(mol, img_dir)

        #         # CHARGE VALUE MAPPING
        #         ndigits = int(request.form["digits"])
        #         _, _, _, charge_dict = charge_value_mapping(gmol, charges, ndigits, img_dir)           
        #         gimg.savefig(gsave_path, bbox_inches="tight")
        #         # charge_img.save(charge_save_path)

        #         return render_template("check_result_gasteiger.html", image=filename, gimage=gfilename, comment=comment, spec=spec)

        #     else:
        #         return render_template("check_result.html", image=filename, comment=comment, spec=spec)
        

        # GASTEIGER CHARGE MAPPING
        if request.form.get("gasteiger") == "On":
            gmol, charges, gimg, gfilename, gsave_path = calc_gasteiger(mol, img_dir)

            # CHARGE VALUE MAPPING
            ndigits = int(request.form["digits"])
            _, _, _, charge_dict = charge_value_mapping(gmol, charges, ndigits, img_dir)           
            gimg.savefig(gsave_path, bbox_inches="tight")
            # charge_img.save(charge_save_path)

            return render_template("check_result_gasteiger.html", image=filename, gimage=gfilename, comment=comment, spec=spec, types_dict=types_dict, charge_dict=charge_dict)

        else:
            return render_template("check_result.html", image=filename, comment=comment, spec=spec, types_dict=types_dict)
    
    else:
        return render_template("check.html", message=None)

############# SIMILARITY CHECK BY SMILES AND INCHI ########################################################

@app.route("/similarity", methods=["GET", "POST"])
def similarity():
    if request.method == "POST":
        if not request.form["smilesA"] or not request.form["smilesB"]:
            return render_template("similarity.html", message=None)

        smilesA = request.form["smilesA"]
        smilesB = request.form["smilesB"]
        num_bits = int(request.form["number"])
        molsPerRow = int(request.form["perrow"])
        
        molA = smiles_or_inchi(smilesA)
        molB = smiles_or_inchi(smilesB)
        
        comment = {}
        comment["MoleculeA"] = smilesA
        comment["MoleculeB"] = smilesB
        
        if molA == None or molB == None:
            generate_message = "Failed to generate mol object."
            return render_template("similarity.html", message=generate_message)

        if os.path.exists("./tempimage"):
            shutil.rmtree("./tempimage")

        img_dir = "./tempimage"
        if not os.path.isdir(img_dir):
            os.mkdir(img_dir)
        
        specA, _, _, _ = calc_specs(molA, img_dir)
        specB, _, _, _ = calc_specs(molB, img_dir)

        # CALC MORGAN FPS, COMMON STRUCTURE IMAGE, BITS AND SIMILARITY COEFFICIENTS
        bits, bitinfoA, bitinfoB, common_bits, common_bit_img, coefs = calc_bits(molA, molB, num_bits, molsPerRow)

        # VISUALIZE ATOM CONTRIBUTIONS TO SIMILARITY
        sim_image, sim_filename, sim_save_path = similarity_map(molA, molB, img_dir)
        sim_image.savefig(sim_save_path, bbox_inches="tight")

        # GRID IMAGE OF MOLECULE_A AND MOLECULE_B
        mols_image, mols_filename, mols_save_path = grid_image(molA, molB, img_dir)
        with open(mols_save_path, "w") as f:
            f.write(mols_image)
            f.close()
        # mols_image.save(mols_save_path)

        return render_template("similarity_result.html", comment=comment, molsimage=mols_filename, bitimage=common_bit_img, simimage=sim_filename, specA=specA, specB=specB, coefs=coefs, bits=bits)

    return render_template("similarity.html")

############# SITE DESCRIPTION ########################################################

@app.route("/about")
def about():
    return render_template('about.html')

########### DASH PLOTLY AREA ###########################################

@app.route("/plotly")
def render():
    return redirect("/sdfplotly")


if __name__ == "__main__":
    app.run(host="0.0.0.0")
