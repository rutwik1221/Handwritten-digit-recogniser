import React, {Component} from 'react';
import * as tf from '@tensorflow/tfjs';
import run from './script.js';
import './Grid.css';
 
let trained = true;

export default class Grid extends Component{
    static defaultProps = {
        nrows : 28,
        ncols : 28,
    }
    constructor(props){
        super(props);
        this.state = {
            grid : [],
            isMouseDown : false,
            prediction:null,
            model : null,
        }
        this.clearGrid=this.clearGrid.bind(this);
        this.pred=this.pred.bind(this);         
    }
    async componentDidMount(){
        const grid=this.getGrid();
        let model = await getModel();
        this.setState({grid:grid,model:model});  
    }
    pred(){ 
        const {model,grid} = this.state;
        let t = tf.tensor(grid);
        t=t.reshape([1,28,28,1]);
        const res = model.predict(t).argMax(-1);
        res.array().then(array => this.setState({prediction:array[0]}));
        t.dispose();
    }
    getGrid() {
        const grid = [];
        for(let row=0;row<this.props.nrows;row++){
            const currentRow = [];
            for(let col=0;col<this.props.ncols;col++){
                currentRow.push(0);
            }
            grid.push(currentRow);
        }
        return grid;
    }
    toggleWall(row, col){
        const newgrid = this.state.grid;
        newgrid[row][col] = 1;
        if(row>0){
            newgrid[row-1][col] = 1;
        }
        if(row<27){
            newgrid[row+1][col] = 1;
        }
        if(col>0){
            newgrid[row][col-1] = 1;
        }
        if(col<27){
            newgrid[row][col+1] = 1;
        }
        if(row>0&&col>0){
            newgrid[row-1][col-1] = 1;
        }
        if(row>0&&col<this.props.ncols-1){
            newgrid[row-1][col+1] = 1;
        }
        if(row<this.props.nrows-1 && col>0){
            newgrid[row+1][col-1] = 1;
        }
        if(row<this.props.nrows-1 && col<this.props.ncols-1){
            newgrid[row+1][col+1] = 1;
        }
        this.setState({grid:newgrid});
    }
    clearGrid(){
        const newgrid = [];
        for(let row=0;row<this.props.nrows;row++){
            const currentRow = [];
            for(let col=0;col<this.props.ncols;col++){
                currentRow.push(0);
            }
            newgrid.push(currentRow);
        }
        this.setState({grid:newgrid,prediction:null})
    }
    handleMouseDown(row, col) {
            this.toggleWall(row, col);
            this.setState({isMouseDown:true});
    }
    handleMouseEnter(row, col) {
        if(this.state.isMouseDown)
            this.toggleWall(row, col);
    }
    handleMouseUp(){
        this.setState({isMouseDown:false});
    }   
    render (){
        const {grid} = this.state;
        let text = (this.state.prediction==null)? "Draw a number in the box":"I think it is a " ;
        return(
            <>
            <div>
                <table>
                    <tbody>
                        {grid.map((row,rowIdx)=>{
                            return(
                                <tr key={rowIdx}>
                                    {row.map((cell,cellIdx)=>{
                                        return (
                                            <td 
                                                key ={`${rowIdx}-${cellIdx}`}
                                                id={`${rowIdx}-${cellIdx}`}
                                                className = {getname(cell)}
                                                onMouseDown = {()=>{
                                                    this.handleMouseDown(rowIdx,cellIdx)
                                                }}
                                                onMouseEnter = {()=>{
                                                    this.handleMouseEnter(rowIdx,cellIdx)
                                                }}
                                                onMouseUp = {()=>{
                                                    this.handleMouseUp(rowIdx,cellIdx)
                                                }}
                                            >         
                                            </td>
                                        )
                                    })}
                                </tr>
                            )
                        })}
                    </tbody>
                </table>
            </div>
            <span>
            <button
                onClick = {this.pred}
            >
                Guess
            </button>
            <button
                onClick = {this.clearGrid}
            >
                Clear Board
            </button>
            <h4>{text}{this.state.prediction}</h4>
            </span>
            
            </>
        );
    }
}

function getname(cell){
    if(cell===1)
        return "on";
    return "off";
}

async function getModel(){
    console.log("Loading Model");
    let model
    try{
        model = await tf.loadLayersModel('localstorage://my-model');
    }
    catch(err){
        trained = false;
        alert("Training the Model, this may take a few minutes\n(This only happens on your first time) ");
        await run();
        model = await tf.loadLayersModel('localstorage://my-model');
    }
    console.log("model Loaded")
    return model;
}
