from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QScrollArea, QWidget


class PreviewModal(QDialog):
    """Show a precomputed PipelinePreview (async) or compute inline for tests."""

    def __init__(self, pipeline=None, dataset=None, parent=None, preview_result=None):
        super().__init__(parent)
        self.setWindowTitle("Pipeline Preview")
        self.setMinimumSize(600, 400)

        self.pipeline = pipeline
        self.dataset = dataset
        self._preview_result = preview_result
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)

        try:
            preview = self._preview_result
            if preview is None:
                if self.pipeline is None or self.dataset is None:
                    raise ValueError("No preview result available.")
                df = self.dataset.dataframe.head(500)
                preview = self.pipeline.preview(df)

            header = QLabel(
                f"<b>Original Shape:</b> {preview.original_shape} → "
                f"<b>Final Shape:</b> {preview.final_shape}"
            )
            content_layout.addWidget(header)

            if preview.total_added_columns or preview.total_removed_columns:
                stats = QLabel(
                    f"Columns added: {preview.total_added_columns}, "
                    f"removed: {preview.total_removed_columns}"
                )
                stats.setObjectName("TextMuted")
                content_layout.addWidget(stats)

            content_layout.addSpacing(10)

            for i, step in enumerate(preview.steps):
                step_title = QLabel(f"<b>Step {i+1}: {step['name']}</b>")
                content_layout.addWidget(step_title)

                shape_lbl = QLabel(f"  Shape: {step['input_shape']} → {step['output_shape']}")
                content_layout.addWidget(shape_lbl)

                if step["added_columns"]:
                    add_lbl = QLabel(
                        f"  + Added: {', '.join(str(c) for c in step['added_columns'])}"
                    )
                    add_lbl.setStyleSheet("color: green;")
                    content_layout.addWidget(add_lbl)

                if step["removed_columns"]:
                    rem_lbl = QLabel(
                        f"  - Removed: {', '.join(str(c) for c in step['removed_columns'])}"
                    )
                    rem_lbl.setStyleSheet("color: red;")
                    content_layout.addWidget(rem_lbl)

                for w in step["warnings"]:
                    w_lbl = QLabel(f"  Warning: {w}")
                    w_lbl.setStyleSheet("color: orange;")
                    content_layout.addWidget(w_lbl)

                content_layout.addSpacing(10)

        except Exception as e:
            err = QLabel(f"Error generating preview:\n{str(e)}")
            err.setObjectName("DangerText")
            content_layout.addWidget(err)

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)
