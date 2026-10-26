"""Views belong to output-block identities; hiding a dock does not remove its block."""
from PySide6.QtCore import Qt
from .plots import CurvePlot, SolutionView

OUTPUT_KINDS = {'monitor', 'solution', 'comparison'}


class OutputViews:
    def __init__(self, window):
        self.window = window
        self.entries = {}

    def open(self, node):
        if node.kind not in OUTPUT_KINDS:
            return
        if node.identifier not in self.entries:
            view = CurvePlot(logarithmic=True) if node.kind == 'monitor' else SolutionView()
            dock = self.window.dock(node.title, view, Qt.DockWidgetArea.BottomDockWidgetArea)
            dock.setObjectName('output-' + node.identifier)
            self.window.tabifyDockWidget(self.window.preview_dock, dock)
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
            sides = self.window.config.sides
            series = [
                ('Total', [(step, row['total']) for step,row in records], '#2b9d91'),
                ('PDE (raw)', [(step, row['pde']) for step,row in records], '#d58b35'),
                ('Mean BC (raw)', [(step,sum(row[s] for s in sides)/len(sides)) for step,row in records], '#7089d5'),
            ] if records else []
            view.set_series(series, 'Requested training losses · log scale · retained updates only' if records else 'Connect this monitor and train to record losses.')
        else:
            data = (self.window.result or {}).get('outputs', {}).get(node.identifier)
            if data is None:
                view.clear()
                view.curve.caption='Connect this output, then train to calculate its data.'
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
