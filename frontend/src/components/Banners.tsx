export interface Banner {
  id: string;
  level: 'error' | 'warn' | 'info';
  text: string;
}

export function Banners({ banners, onDismiss }: { banners: Banner[]; onDismiss: (id: string) => void }) {
  if (banners.length === 0) return null;
  return (
    <div className="banners" role="region" aria-label="Notifications">
      {banners.map((b) => (
        <div key={b.id} className={`banner banner-${b.level}`} role={b.level === 'error' ? 'alert' : 'status'}>
          <span className="banner-dot" aria-hidden="true" />
          <span className="banner-text">{b.text}</span>
          <button className="icon-btn small" onClick={() => onDismiss(b.id)} aria-label="Dismiss">
            ×
          </button>
        </div>
      ))}
    </div>
  );
}
