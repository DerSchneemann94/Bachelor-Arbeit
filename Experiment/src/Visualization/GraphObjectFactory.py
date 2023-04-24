import plotly.graph_objects as go

class GraphObjectFactory:

    def __init__(self) -> None:
        self.color = ["red", "orange", "blue","green","yellow", "gray","black"]
        self.color_state = {}


    def create_bar_object(self, x_data, y_data, name, meta):
        color = self.__get_and_adjust_color("bar")
        bar = go.Bar(
            x=x_data,
            y=y_data,
            name=name,
            marker_color=color,
            meta=meta,
        )
        return bar
 

    def __get_and_adjust_color(self, graph_type): 
        graph_types = list(self.color_state.keys())
        if graph_type in graph_types:
            self.color_state[graph_type] = self.color_state[graph_type] + 1 
        else:
            self.color_state[graph_type] = 0
        return self.color[self.color_state[graph_type]]    
    


    