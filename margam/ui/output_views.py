"""Views belong to output-block identities; hiding a dock does not remove its block."""
from PySide6.QtCore import Qt
from .plots import CurvePlot, SolutionView
from PySide6.QtWidgets import QWidget,QVBoxLayout,QLabel,QTableWidget,QTableWidgetItem

OUTPUT_KINDS = {'monitor', 'solution', 'comparison', 'export', 'paraview'}


class VectorResultView(QWidget):
    def __init__(self):
        super().__init__();layout=QVBoxLayout(self)
        self.caption=QLabel('Connect and train to calculate vector fields.');self.caption.setWordWrap(True);layout.addWidget(self.caption)
        self.table=QTableWidget();self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers);layout.addWidget(self.table)

    def clear(self):
        self.table.clear();self.table.setRowCount(0);self.caption.setText('Model changed. Train again or open a saved run.')

    def show_result(self,data):
        fields=data['fields'];points=data['points'];reference=data.get('reference')
        headers=['x','y',*fields]+(['ref '+f for f in fields]+['error '+f for f in fields] if reference is not None else [])
        count=min(200,len(points));self.table.setColumnCount(len(headers));self.table.setHorizontalHeaderLabels(headers);self.table.setRowCount(count)
        for index in range(count):
            row=[*points[index],*data['values'][index]]
            if reference is not None:row.extend(reference[index]);row.extend(a-b for a,b in zip(data['values'][index],reference[index]))
            for column,value in enumerate(row):self.table.setItem(index,column,QTableWidgetItem(f'{value:.7g}'))
        self.table.resizeColumnsToContents()
        caption=f"Fields {', '.join(fields)} · first {count} of {len(points)} rows · {data['steps']} updates"
        if data.get('rmse_by_field'):caption+=' · RMSE '+', '.join(name+'='+f'{value:.4g}' for name,value in data['rmse_by_field'].items())
        self.caption.setText(caption)


class OutputViews:
    def __init__(self, window):
        self.window = window
        self.entries = {}

    def open(self, node):
        if node.kind not in OUTPUT_KINDS or node.kind in ('export','paraview'):
            return
        if node.identifier not in self.entries:
            view = CurvePlot(logarithmic=True) if node.kind == 'monitor' else VectorResultView() if len(self.window.config.field_names)>1 else SolutionView()
            dock = self.window.dock(node.title, view, Qt.DockWidgetArea.BottomDockWidgetArea)
            dock.setObjectName('output-' + node.identifier)
            for other,_ in self.entries.values():
                if not other.isFloating():
                    hidden=other.isHidden()
                    self.window.tabifyDockWidget(other, dock)
                    if hidden:other.hide()
                    break
            self.entries[node.identifier] = (dock, view)
            self.window.view_menu.addAction(dock.toggleViewAction())
        self.refresh(node)
        dock, _ = self.entries[node.identifier]
        dock.setWindowTitle(node.title)
        dock.show()
        dock.raise_()
        return dock

    def refresh(self, node):
        entry = self.entries.get(node.identifier)
        if entry is None:
            return
        _, view = entry
        if node.kind == 'monitor':
            records = self.window.monitor_records.get(node.identifier, [])
            sides = tuple(name for name in self.window.config.constraint_names if name!='pressure_gauge')
            series = [
                ('Total', [(step, row['total']) for step,row in records], '#2b9d91'),
                ('PDE (raw)', [(step, row['pde']) for step,row in records], '#d58b35'),
                ('Mean BC (raw)', [(step,sum(row[s] for s in sides)/len(sides)) for step,row in records], '#7089d5'),
            ] if records else []
            if records and self.window.config.physics=='incompressible_2d':series.append(('Pressure gauge (raw)',[(step,row['pressure_gauge']) for step,row in records],'#ad75ce'))
            view.set_series(series, 'Requested training losses · log scale · retained updates only' if records else 'Connect this monitor and train to record losses.')
        else:
            data = (self.window.result or {}).get('outputs', {}).get(node.identifier)
            vector=len(data.get('fields',[]))>1 if data is not None else len(self.window.config.field_names)>1
            if vector!=isinstance(view,VectorResultView):
                dock,old=entry;hidden=dock.isHidden();view=VectorResultView() if vector else SolutionView()
                dock.setWidget(view);old.deleteLater();self.entries[node.identifier]=(dock,view)
                if hidden:dock.hide()
            if data is None:
                view.clear()
                if hasattr(view,'curve'):view.curve.caption='Connect this output, then train to calculate its data.'
            else:
                view.show_result(data)

    def clear_data(self):
        for _,view in self.entries.values():
            if isinstance(view, CurvePlot):
                view.set_series([], 'Model changed. Run training again.')
            else:
                view.clear()

    def sync(self, nodes):
        identifiers = {node.identifier for node in nodes}
        for identifier in list(self.entries):
            if identifier not in identifiers:
                dock,_ = self.entries.pop(identifier)
                dock.close()
                self.window.removeDockWidget(dock)
                dock.deleteLater()
