from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union

class LabelComponent(BaseModel):
    type: Literal["label"] = "label"
    text: str
    fontSize: int

class ScrollTextComponent(BaseModel):
    type: Literal["scrollText"] = "scrollText"
    text: str

class ButtonComponent(BaseModel):
    type: Literal["button"] = "button"
    text: str
    cta: str

class DetailCardComponent(BaseModel):
    type: Literal["detailCard"] = "detailCard"
    title: str
    text: str
    date: str
    time: str
    buttonTitle: str
    cta: str

class CompositeCardComponent(BaseModel):
    type: Literal["compositeCard"] = "compositeCard"
    text: str
    buttonTitle: str
    cta: str

class GraphDataPoint(BaseModel):
    x: float
    y: float

class GraphComponent(BaseModel):
    type: Literal["graph"] = "graph"
    graphType: Literal["line", "bar"]
    title: str
    xAxisLabels: List[str]
    yAxisLabels: List[str]
    dataPoints: List[GraphDataPoint]

class PieChartEntry(BaseModel):
    label: str
    value: float

class PieChartComponent(BaseModel):
    type: Literal["pieChart"] = "pieChart"
    centerText: str
    entries: List[PieChartEntry]

# Union of all component types
UIComponent = Union[
    LabelComponent,
    ScrollTextComponent,
    ButtonComponent,
    DetailCardComponent,
    CompositeCardComponent,
    GraphComponent,
    PieChartComponent
]

class UIComponentsResponse(BaseModel):
    components: List[UIComponent] 