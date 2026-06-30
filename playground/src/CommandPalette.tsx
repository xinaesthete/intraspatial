// A `/`-triggered command palette for inserting a node at the cursor. Type to filter,
// ↑/↓ to move, ↵ to insert the highlighted node, Esc to close. Designed to grow more
// commands later (the items list is the only node-specific part).
import { useEffect, useMemo, useRef, useState } from "react";

export interface CmdItem {
  name: string;
  label: string;
  category: string;
  describe?: string;
}

export function CommandPalette({
  items,
  onPick,
  onClose,
}: {
  items: CmdItem[];
  onPick: (name: string) => void;
  onClose: () => void;
}) {
  const [q, setQ] = useState("");
  const [idx, setIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return items;
    return items.filter(
      (it) =>
        it.label.toLowerCase().includes(s) ||
        it.name.toLowerCase().includes(s) ||
        it.category.toLowerCase().includes(s),
    );
  }, [q, items]);

  useEffect(() => {
    setIdx(0);
  }, [q]);

  // Keep the highlighted row in view.
  useEffect(() => {
    const el = listRef.current?.querySelector<HTMLElement>(`[data-i="${idx}"]`);
    el?.scrollIntoView({ block: "nearest" });
  }, [idx]);

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setIdx((i) => Math.min(i + 1, filtered.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setIdx((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      const it = filtered[idx];
      if (it) onPick(it.name);
    } else if (e.key === "Escape") {
      e.preventDefault();
      onClose();
    }
  };

  return (
    <div className="cmd-overlay" onMouseDown={onClose}>
      <div className="cmd-panel" onMouseDown={(e) => e.stopPropagation()}>
        <input
          ref={inputRef}
          className="cmd-input"
          placeholder="Insert node…   type to filter · ↑↓ · ↵ · esc"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={onKeyDown}
        />
        <div className="cmd-list" ref={listRef}>
          {filtered.length === 0 && <div className="cmd-empty">No matching nodes</div>}
          {filtered.map((it, i) => (
            <button
              key={it.name}
              data-i={i}
              className={`cmd-item${i === idx ? " active" : ""}`}
              onMouseEnter={() => setIdx(i)}
              onClick={() => onPick(it.name)}
            >
              <span className="cmd-label">{it.label}</span>
              <span className="cmd-cat">{it.category}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
